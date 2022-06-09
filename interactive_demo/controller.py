import torch
import numpy as np
from tkinter import messagebox

from isegm.data.interaction_generators import PointGenerator, ContourGenerator, StrokeGenerator, AxisTransformType
from isegm.data.interaction_type import IType
from isegm.inference.clicker import Clicker, Click
from isegm.inference.compose_aggregator import ComposeAggregator
from isegm.inference.contour_aggregator import ContourAggregator
from isegm.inference.predictors import get_predictor
from isegm.inference.stroke_aggregator import StrokeAggregator
from isegm.utils.vis import draw_with_blend_and_clicks, draw_with_blend_and_lines


class InteractiveController:
    def __init__(
        self, net, device,
        predictor_params, update_image_callback,
        prob_thresh=0.5, model_input_type=IType.point,
        contour_filled=True
    ):
        self.net = net
        self.model_input_type = model_input_type
        self.prob_thresh = prob_thresh
        interactive_imitators_dict = {
            k: v
            for k, v in {
                IType.point: Clicker(generator=PointGenerator(
                    deterministic=True, at_max_mask=True, sfc_inner_k=-1, fit_normal=True
                )),
                IType.contour: ContourAggregator(generator=ContourGenerator(
                    deterministic=True, one_component=True, convex=True, width=10, filled=contour_filled
                )),
                IType.stroke: StrokeAggregator(generator=StrokeGenerator(
                    deterministic=True, one_component=True, axis_transform=AxisTransformType.sine
                ))
            }.items() if k in self.model_input_type
        }
        self.interactive_imitator = ComposeAggregator(interactive_imitators_dict, exclude_reset=[])
        self.input_type = None
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.interactive_imitator.set_index_offset(1)

    def add_interaction(self, **kwargs):
        im_prev_state = self.interactive_imitator.get_state(self.input_type)
        pred_prev_state = self.predictor.get_states()

        if self.input_type == IType.point:
            click = Click(is_positive=kwargs['is_positive'], coords=(kwargs['y'], kwargs['x']))
            self.interactive_imitator[self.input_type].add_point(click)
        else:
            self.interactive_imitator[self.input_type].finish()

        self.states.append({
            'imitator': im_prev_state,
            'predictor': pred_prev_state
        })
        pred = self.predictor.get_prediction(
            self.interactive_imitator, prev_mask=self._init_mask, input_type=self.input_type
        )
        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def update_interaction(self, **kwargs):
        self.interactive_imitator[self.input_type].add_point((kwargs['y'], kwargs['x']))

    def begin_interaction(self, **kwargs):
        self.interactive_imitator[self.input_type].begin(kwargs['is_positive'], [(kwargs['y'], kwargs['x'])])

    def undo_last(self):
        if not self.states:
            return
        prev_state = self.states.pop()
        self.interactive_imitator.set_state(prev_state['imitator'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.interactive_imitator.reset()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.interactive_imitator.set_index_offset(0)

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask

    def get_visualization(self, alpha_blend, click_radius, line_width):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        interactions_dict = self.interactive_imitator.get_interactions()
        if self.input_type == IType.point:
            vis = draw_with_blend_and_clicks(
                self.image, results_mask_for_vis, clicks_list=interactions_dict[self.input_type],
                alpha=alpha_blend, radius=click_radius
            )
            if self.probs_history:
                total_mask = self.probs_history[-1][0] > self.prob_thresh
                results_mask_for_vis[np.logical_not(total_mask)] = 0
                vis = draw_with_blend_and_clicks(
                    vis, mask=results_mask_for_vis,
                    alpha=alpha_blend, radius=click_radius
                )
        else:
            lines_list = interactions_dict[self.input_type]
            vis = draw_with_blend_and_lines(
                self.image, mask=results_mask_for_vis,
                interactions_list=lines_list, alpha=alpha_blend, thickness=line_width
            )
        return vis
