from collections import defaultdict
import logging
from time import time

import cv2
import numpy as np
import torch

from isegm.data.interaction_generators import PointGenerator, ContourGenerator, StrokeGenerator, AxisTransformType
from isegm.data.interaction_type import IType, ProductInteractionSelector
from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.inference.compose_aggregator import ComposeAggregator
from isegm.inference.contour_aggregator import ContourAggregator
from isegm.inference.stroke_aggregator import StrokeAggregator
from isegm.utils.log import TqdmToLogger
from isegm.utils.misc import limit_longest_size

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, interaction_selector, logger=None, **kwargs):
    n_samples = len(dataset)
    if logger is None:
        tbar = tqdm(range(n_samples), leave=False)
    else:
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        tbar = tqdm(range(n_samples), file=tqdm_out, ncols=100)
    all_ious = []
    all_times = []
    start_time = time()
    for index in tbar:
        if logger is not None:
            tbar.set_description(f'{dataset.name}')
        sample = dataset.get_sample(index)
        gt_masks = sample.gt_masks
        for gt_mask in gt_masks:
            if isinstance(interaction_selector, ProductInteractionSelector):
                interaction_selector.reset()
                sample_ious = []
                sample_times = {}
                for i in range(len(interaction_selector)):
                    _, prod_ious, _, prod_times = evaluate_sample(
                        sample.image, gt_mask, predictor,
                        sample_id=index, interaction_selector=interaction_selector,
                        **kwargs
                    )
                    interaction_selector.next_product()
                    sample_ious.append(prod_ious)
                    for k, v in prod_times.items():
                        sample_times[k] = sample_times.get(k, 0) + v
            else:
                _, sample_ious, _, sample_times = evaluate_sample(
                    sample.image, gt_mask, predictor,
                    sample_id=index, interaction_selector=interaction_selector,
                    **kwargs
                )
            if isinstance(interaction_selector, ProductInteractionSelector):
                for k in sample_times:
                    sample_times[k] /= len(interaction_selector)
            all_ious.append(sample_ious)
            all_times.append(sample_times)
    end_time = time()
    elapsed_time = end_time - start_time
    return all_ious, elapsed_time, all_times


def evaluate_sample(
    image, gt_mask, predictor, max_iou_thr,
    pred_thr=0.49, min_interactions=1, max_interactions=20,
    sample_id=None, callback=None, interaction_selector=None,
    boundary_iou=False, boundary_size=0.02, accumulate_interactions=False,
    contour_filled=True, max_size=None,
):
    orig_gt_mask = gt_mask.copy()
    if max_size is not None:
        image, new_size = limit_longest_size(image, max_size)
        gt_mask, new_size = limit_longest_size(gt_mask, None, new_size, cv2.INTER_NEAREST)
    interactive_imitators_dict = {
        k: v
        for k, v in {
            IType.point: Clicker(gt_mask=gt_mask, generator=PointGenerator(
                at_max_mask=True, sfc_inner_k=-1, fit_normal=True
            )),
            IType.contour: ContourAggregator(gt_mask=gt_mask, generator=ContourGenerator(
                one_component=True, convex=True, width=10, filled=contour_filled
            )),
            IType.stroke: StrokeAggregator(gt_mask=gt_mask, generator=StrokeGenerator(
                one_component=True, axis_transform=AxisTransformType.sine
            ))
        }.items() if k in interaction_selector.itypes
    }
    interactive_imitator = ComposeAggregator(interactive_imitators_dict, exclude_reset=[IType.point])
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    times_dict = defaultdict(float)

    with torch.no_grad():
        predictor.set_input_image(image)
        interaction_i = 0
        for interaction_i in range(max_interactions):
            input_type = interaction_selector.select_itype(interaction_i, pred_mask)
            if not accumulate_interactions:
                interactive_imitator.reset()
            start = time()
            interactive_imitator.make_next(pred_mask, input_type=input_type)
            end = time()
            times_dict['interaction'] += end - start

            prev_mask = predictor.prev_prediction.cpu().numpy()[0, 0]
            start = time()
            pred_probs = predictor.get_prediction(interactive_imitator, input_type=input_type)
            pred_mask = pred_probs > pred_thr
            end = time()
            times_dict['prediction'] += end - start

            if callback is not None:
                callback(
                    image, gt_mask, prev_mask, pred_probs,
                    sample_id, interaction_i, interactive_imitator.interactions_dict,
                    input_type
                )

            start = time()
            if max_size is not None:
                pred_mask_resized, new_size = limit_longest_size(
                    pred_mask, None, orig_gt_mask.shape[:2],
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                pred_mask_resized = pred_mask
            if boundary_iou:
                iou = utils.get_boundary_iou(orig_gt_mask, pred_mask_resized, dilation_ratio=boundary_size)
            else:
                iou = utils.get_iou(orig_gt_mask, pred_mask_resized)
            ious_list.append(iou)
            end = time()
            times_dict['iou'] += end - start

            if iou >= max_iou_thr and interaction_i + 1 >= min_interactions:
                break
        for k in times_dict:
            times_dict[k] = times_dict[k] / (interaction_i + 1)
        return interactive_imitators_dict, np.array(ious_list, dtype=np.float32), pred_probs, times_dict
