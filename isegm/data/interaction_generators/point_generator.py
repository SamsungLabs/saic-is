from functools import lru_cache
import math

import cv2
import numpy as np

from isegm.utils.vis import draw_points
from .base import BaseInteractionGenerator, remove_thin_border


class PointGenerator(BaseInteractionGenerator):
    def __init__(
        self,
        max_points=1, prob_gamma=0.8,
        sfc_inner_k=-1, first_click_center=True,
        at_max_mask=False, fit_normal=False,
        sigma_ratio=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_points = max_points
        self.sfc_inner_k = sfc_inner_k
        self._pos_probs = generate_probs(self.max_points, gamma=prob_gamma)
        self._neg_probs = generate_probs(self.max_points, gamma=prob_gamma)
        self.at_max_mask = at_max_mask
        self.fit_normal = fit_normal
        self._sigma_ratio = sigma_ratio
        self.first_click_center = first_click_center

    def generate_points(self, mask, is_positive=True, num_points=None):
        generator = self.get_random_generator(mask, is_positive)
        if num_points is None:
            if is_positive:
                num_points = 1 + generator.choice(np.arange(self.max_points), p=self._pos_probs)
            else:
                num_points = generator.choice(np.arange(self.max_points), p=self._neg_probs)

        indices_probs = None
        if isinstance(mask, (list, tuple)):
            indices_probs = [x[1] for x in mask]
            indices = [(np.argwhere(x), prob) for x, prob in mask]
            if indices_probs:
                assert math.isclose(sum(indices_probs), 1.0)
        elif self.at_max_mask:
            indices = np.argwhere(mask == mask.max())
        else:
            indices = np.argwhere(mask)

        points = []
        for j in range(num_points):
            first_click = self.first_click_center and j == 0 and indices_probs is None

            if first_click and mask.sum() > 0:
                point_indices = get_point_candidates(
                    mask, k=self.sfc_inner_k,
                    fit_normal=self.fit_normal, sigma_ratio=self._sigma_ratio,
                    generator=generator,
                )
            elif indices_probs:
                point_indices_indx = generator.choice(np.arange(len(indices)), p=indices_probs)
                point_indices = indices[point_indices_indx][0]
            else:
                point_indices = indices

            num_indices = len(point_indices)
            if num_indices > 0:
                point_indx = 0 if first_click else 100
                click = point_indices[generator.randint(0, num_indices)].tolist() + [point_indx]
                points.append(click)

        return points

    def generate_points_mask(self, mask, is_positive, num_points=None, radius=3):
        points = self.generate_points(mask, is_positive, num_points)
        points_mask = draw_points(np.zeros_like(mask), points, [255, 255, 255], radius=radius)
        return points_mask


def get_point_candidates(obj_mask, k=1.7, fit_normal=False, sigma_ratio=1, generator=np.random):
    reduced_mask = remove_thin_border(obj_mask.astype(np.uint8), dilation_ratio=0.001)
    padded_mask = np.pad(reduced_mask, ((1, 1), (1, 1)), 'constant')
    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
    if k > 0:
        inner_mask = dt > dt.max() / k
        return np.argwhere(inner_mask)
    else:
        mask_dt_indices = np.argwhere(dt > 0)
        if len(mask_dt_indices) == 0:
            return np.argwhere(obj_mask)
        prob_map = dt[dt > 0].flatten()
        prob_map /= max(prob_map.sum(), 1e-6)
        if fit_normal:
            pmax = prob_map.max()
            pmin = prob_map.min()
            mu = pmax
            sigma = (pmax - pmin) / sigma_ratio
            random_distance = generator.normal(mu, sigma, 1)
            # generate only numbers within [mu - sigma * sigma_ratio; mu + sigma * sigma_ratio]
            while random_distance < pmin or mu + (mu - pmin) < random_distance:
                random_distance = generator.normal(mu, sigma, 1)
            if random_distance > mu:
                right_shift = random_distance - mu
                random_distance = mu - right_shift
            distance_map = np.abs(prob_map - random_distance)
            # np.argmin returns only first occurrence
            click_indx = np.argwhere(distance_map == distance_map.min()).ravel()
            click_indx = generator.choice(click_indx)
        else:
            click_indx = generator.choice(range(len(prob_map)), p=prob_map)
        click_coords = mask_dt_indices[click_indx]
        return np.array([click_coords])


@lru_cache(maxsize=None)
def generate_probs(max_num_points, gamma):
    probs = []
    last_value = 1
    for i in range(max_num_points):
        probs.append(last_value)
        last_value *= gamma

    probs = np.array(probs, dtype=np.float64)
    probs /= probs.sum()

    return probs
