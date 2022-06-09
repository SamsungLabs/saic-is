import random

import numpy as np

from isegm.data.interaction_generators import PointGenerator
from isegm.data.interaction_samplers.base import BaseInteractionSampler


class MultiPointSampler(BaseInteractionSampler):
    def __init__(
        self,
        generator=PointGenerator(),
        only_one_first_click=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.only_one_first_click = only_one_first_click

    def sample_interaction(self):
        assert self._selected_mask is not None
        pos_points = self._multi_mask_sample_points(
            self._selected_masks, is_positive=[True] * len(self._selected_masks)
        )

        if np.random.rand() < self.neg_prob:
            neg_strategy = [(self._neg_masks[k], prob)
                            for k, prob in zip(self.neg_strategies, self.neg_strategies_prob)]
            neg_masks = self._neg_masks['required'] + [neg_strategy]
            neg_points = self._multi_mask_sample_points(
                neg_masks,
                is_positive=[True] * len(self._neg_masks['required']) + [False]
            )
        else:
            neg_points = [(-1, -1, -1)] * self.max_objects
        return pos_points + neg_points

    def _multi_mask_sample_points(self, selected_masks, is_positive):
        selected_masks = selected_masks[:self.max_objects]

        each_obj_points = [
            self.generator.generate_points(mask, is_positive=is_positive[i])
            for i, mask in enumerate(selected_masks)
        ]
        each_obj_points = [x for x in each_obj_points if len(x) > 0]

        points = []
        if len(each_obj_points) == 1:
            points = each_obj_points[0]
        elif len(each_obj_points) > 1:
            if self.only_one_first_click:
                each_obj_points = each_obj_points[:1]

            points = [obj_points[0] for obj_points in each_obj_points]

            aggregated_masks_with_prob = []
            for indx, x in enumerate(selected_masks):
                if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
                    for t, prob in x:
                        aggregated_masks_with_prob.append((t, prob / len(selected_masks)))
                else:
                    aggregated_masks_with_prob.append((x, 1.0 / len(selected_masks)))

            other_points_union = self.generator.generate_points(aggregated_masks_with_prob, is_positive=False)
            if len(other_points_union) + len(points) <= self.max_objects:
                points.extend(other_points_union)
            else:
                points.extend(random.sample(other_points_union, self.max_objects - len(points)))

        if len(points) < self.max_objects:
            points.extend([(-1, -1, -1)] * (self.max_objects - len(points)))

        return points
