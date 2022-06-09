import numpy as np

from isegm.data.interaction_generators import ContourGenerator
from isegm.utils.misc import get_bbox_from_mask, get_iou
from .base import BaseInteractionSampler


class MultiContourSampler(BaseInteractionSampler):
    def __init__(self, generator=ContourGenerator(), **kwargs):
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
            contours_mask = np.tile(np.zeros_like(self.selected_mask, dtype=np.float32), (2, 1, 1))
        else:
            contours_mask = self.generator.generate_contour_mask(pos_mask, is_positive=True)

        if np.random.rand() < self.neg_prob and pos_mask is not None:
            neg_strategy = list(zip(*[
                (self._neg_masks[k], prob, k)
                for k, prob in zip(self.neg_strategies, self.neg_strategies_prob)
                if prob > 0
            ]))
            neg_i = np.random.choice(range(len(neg_strategy[0])), p=neg_strategy[1])
            neg_mask = neg_strategy[0][neg_i]
            bbox = get_bbox_from_mask(pos_mask)
            zero_dim = int(np.random.rand() < 0.5)
            left, right = bbox[zero_dim * 2:zero_dim * 2 + 2]
            bounds = int(left * 1.1), int(right * 0.9)
            if zero_dim == 0:
                neg_mask[bounds[0]:bounds[1]] = 0
            else:
                neg_mask[:, bounds[0]:bounds[1]] = 0
            neg_contours_mask = self.generator.generate_contour_mask(neg_mask, is_positive=False)
            if get_iou(pos_mask, neg_contours_mask[1], gt_relative=True) < 0.05:
                contours_mask = contours_mask | neg_contours_mask

        return contours_mask
