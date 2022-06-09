import cv2
import numpy as np

from isegm.data.contour import Contour
from isegm.data.interaction_generators.mask_transforms import (
    leave_one_component,
    random_convex_hull,
    random_deformations, random_dilate_or_erode, random_elastic_transform,
    remove_holes,
    remove_small_components
)
from .base import BaseInteractionGenerator


class ContourGenerator(BaseInteractionGenerator):
    def __init__(self, convex=False, filled=True, width=10, shrink=True, **kwargs):
        super().__init__(**kwargs)
        self.convex = convex
        self.filled = filled
        self.width = width
        self.shrink = shrink

    def generate_contour(self, mask, is_positive):
        generator = self.get_random_generator(mask, is_positive)

        closed_mask = mask.copy()
        closed_mask = remove_holes(closed_mask)
        closed_mask = random_dilate_or_erode(
            closed_mask, generator=generator,
            dilate_scale_range=(16., 32.),
            dilate_prob=0.5 if is_positive else 0,
            erode_scale_range=(24., 64.), min_erode_ratio=0.4,
            shrink_to_object=self.shrink
        )
        if self.one_component:
            closed_mask, centroid = leave_one_component(closed_mask)
        else:
            closed_mask = remove_small_components(closed_mask, small_thresh=0.05)
        if self.convex and is_positive:
            closed_mask, is_convex = random_convex_hull(closed_mask, convex_prob=0.5)
        # Elastic transform may lead to ragged edges, therefore
        # apply elastic transform before random smoothing and single component extraction.
        image_area = mask.shape[0] * mask.shape[1]
        if image_area > 0 and closed_mask.sum() / image_area > 1 / 64.:
            closed_mask = random_elastic_transform(closed_mask, generator=generator, alpha_range=(0.8, 1.2),
                                                   alpha_affine_range=(0.4, 0.6), sigma_range=(0.5, 0.75),
                                                   shrink_to_object=self.shrink)
        if self.one_component:
            closed_mask, centroid = leave_one_component(closed_mask)
        else:
            closed_mask = remove_small_components(closed_mask, small_thresh=0.1)

        closed_mask = random_deformations(
            closed_mask, self.shrink,
            decreased_shift=~is_positive, generator=generator,
        )

        if self.one_component:
            largest_component, centroid = leave_one_component(closed_mask)
            if centroid is None:
                return [Contour(is_positive, coords=None)]
        else:
            largest_component = closed_mask

        if largest_component.sum() < 10:
            return [Contour(is_positive, coords=None)]

        contours, hierarchy = cv2.findContours(
            largest_component.astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        result = []
        for i in range(len(contours)):
            coords = contours[i][:, 0]
            coords = [(y, x) for x, y in coords]
            result.append(Contour(is_positive, coords=coords))

        if len(result) == 0:
            return [Contour(is_positive, coords=None)]
        if len(result) > 1 and self.one_component:
            return [result[0]]

        return result

    def generate_contour_mask(self, mask, is_positive):
        contours = self.generate_contour(mask > 0, is_positive=is_positive)
        contours_mask = Contour.get_mask(contours, mask.shape, self.filled, thickness=self.width)
        return contours_mask
