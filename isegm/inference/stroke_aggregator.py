from copy import deepcopy

import numpy as np

from isegm.data.interaction_generators import StrokeGenerator
from isegm.data.interaction_type import IType
from isegm.data.stroke import Stroke
from isegm.inference.utils import get_fn_fp_distance_transform


class StrokeAggregator:
    def __init__(
        self, gt_mask=None, init_strokes=None,
        ignore_label=-1, stroke_indx_offset=0,
        generator=StrokeGenerator(deterministic=True, one_component=True)
    ):
        self.stroke_indx_offset = stroke_indx_offset
        self.current_stroke = None
        self.generator = generator
        self.input_type = IType.stroke
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset()

        if init_strokes is not None:
            for stroke in init_strokes:
                self.add_stroke(stroke)

    def make_next(self, pred_mask, **kwargs):
        assert self.gt_mask is not None
        stroke = self._get_next_stroke(pred_mask)
        self.add_stroke(stroke)

    def get_interactions(self, strokes_limit=None):
        ans = self.interactions_list[:strokes_limit]
        if self.current_stroke is not None:
            ans.append(self.get_current())
        return ans

    def get_last_stroke(self):
        return self.interactions_list[-1] if len(self.interactions_list) else None

    def _get_next_stroke(self, pred_mask, padding=True):
        fn_mask, fn_mask_dt, fp_mask, fp_mask_dt = get_fn_fp_distance_transform(
            pred_mask, self.gt_mask,
            self.not_clicked_map, self.not_ignore_mask, padding
        )
        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)
        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            mask = fn_mask
        else:
            mask = fp_mask
        stroke = self.generator.generate_stroke(mask, is_positive=is_positive)
        return stroke

    def begin(self, is_positive, coords=None):
        self.current_stroke = Stroke(is_positive, coords)

    def add_point(self, coords):
        self.current_stroke.add_point(coords)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def get_current(self):
        return self.current_stroke

    def finish(self):
        if self.current_stroke is not None:
            self.add_stroke(self.get_current())
        self.current_stroke = None

    def add_stroke(self, stroke):
        stroke.indx = self.stroke_indx_offset + self.num_pos_strokes + self.num_neg_strokes
        if stroke.is_positive:
            self.num_pos_strokes += 1
        else:
            self.num_neg_strokes += 1
        self.interactions_list.append(stroke)

    def _remove_last_click(self):
        stroke = self.interactions_list.pop()

        if stroke.is_positive:
            self.num_pos_strokes -= 1
        else:
            self.num_neg_strokes -= 1

        if self.gt_mask is not None:
            for c in stroke.coords:
                self.not_clicked_map[c[0], c[1]] = True

    def reset(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)
        self.num_pos_strokes = 0
        self.num_neg_strokes = 0
        self.interactions_list = []

    def get_state(self):
        return deepcopy(self.interactions_list)

    def set_state(self, state):
        self.reset()
        for stroke in state:
            self.add_stroke(stroke)

    def set_index_offset(self, index_offset):
        self.stroke_indx_offset = index_offset

    def __len__(self):
        return len(self.interactions_list)
