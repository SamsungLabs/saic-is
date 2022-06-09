import numpy as np

from isegm.data.interaction_generators import StrokeGenerator
from isegm.utils.misc import get_bbox_from_mask, get_iou
from .base import BaseInteractionSampler


class MultiStrokesSampler(BaseInteractionSampler):
    def __init__(self, generator=StrokeGenerator(), **kwargs):
        super().__init__(**kwargs)
        self.generator = generator

    def sample_interaction(self):
        assert self._selected_mask is not None
        selected_masks = self._selected_masks[:self.max_objects]
        pos_mask = None
        for mask in selected_masks:
            if len(mask) == 0:
                continue
            if pos_mask is None:
                pos_mask = mask.copy()
            else:
                pos_mask = np.logical_or(pos_mask, mask)
        if pos_mask is None:
            stroke = np.tile(np.zeros_like(self.selected_mask, dtype=np.float32), (2, 1, 1))
        else:
            pos_mask = pos_mask.astype(np.uint8)
            stroke = self.generator.generate_stroke_mask(pos_mask, is_positive=True)

        if np.random.rand() < self.neg_prob and pos_mask is not None:
            neg_strategy = list(zip(*[
                (self._neg_masks[k], prob)
                for k, prob in zip(self.neg_strategies, self.neg_strategies_prob)
                if prob > 0
            ]))
            neg_i = np.random.choice(range(len(neg_strategy[0])), p=neg_strategy[1])
            neg_mask = neg_strategy[0][neg_i].astype(np.uint8)
            bbox = get_bbox_from_mask(pos_mask)
            zero_dim = int(np.random.rand() < 0.5)
            left, right = bbox[zero_dim * 2:zero_dim * 2 + 2]
            bounds = int(left * 1.1), int(right * 0.9)
            if zero_dim == 0:
                neg_mask[bounds[0]:bounds[1]] = 0
            else:
                neg_mask[:, bounds[0]:bounds[1]] = 0
            neg_stroke = self.generator.generate_stroke_mask(neg_mask, is_positive=False)
            if get_iou(neg_stroke[1], pos_mask, gt_relative=True) < 0.05:
                stroke = stroke | neg_stroke
        return stroke
