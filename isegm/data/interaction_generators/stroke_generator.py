import enum
import warnings

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from isegm.data.interaction_generators.mask_transforms import leave_one_component, remove_small_components
from isegm.data.stroke import Stroke
from isegm.utils.misc import get_bbox_from_mask, get_boundary_size, mask_to_boundary
from .base import BaseInteractionGenerator, remove_thin_border


class AxisTransformType(enum.Enum):
    identity = 'identity'
    sine = 'sine'


class IdentityAxisTransform:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


class SineAxisTransform(IdentityAxisTransform):
    def __init__(self, axis_size, min_period_ratio=0.3, generator=np.random):
        super().__init__()
        self.axis_size = axis_size
        self.generator = generator
        self.min_period_ratio = min_period_ratio
        self.period_coeff = (
            self.generator.rand() * (1 - self.min_period_ratio) + self.min_period_ratio
        ) * self.axis_size

    def __call__(self, x):
        return np.sin(x / (self.period_coeff * np.pi))


class StrokeGenerator(BaseInteractionGenerator):
    def __init__(
        self,
        width=10, max_degree=4, points_count=300,
        axis_transform=AxisTransformType.identity,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.width = width
        self.max_degree = max_degree
        self.axis_transform = (
            IdentityAxisTransform
            if axis_transform == AxisTransformType.identity else
            SineAxisTransform
        )
        self.points_count = points_count

    def generate_stroke(self, mask, is_positive, with_meta=False):
        meta = {
            'basis': ([], []),
            'ends': [-1] * 4,
            'poly': None,
            'transform': None,
        }
        stroke = Stroke(is_positive)

        inner_mask = remove_thin_border(mask, dilation_ratio=0.001)
        if self.one_component:
            inner_mask, centroid = leave_one_component(inner_mask)
        else:
            inner_mask = remove_small_components(inner_mask)
        if inner_mask.sum() == 0:
            if with_meta:
                return stroke, meta
            return stroke

        generator = self.get_random_generator(inner_mask > 0, is_positive)
        indices = np.argwhere(inner_mask)
        if len(indices) < 2:
            if with_meta:
                return stroke, meta
            return stroke

        bbox = get_bbox_from_mask(inner_mask > 0)
        bbox_h = bbox[1] - bbox[0] + 1
        bbox_w = bbox[3] - bbox[2] + 1

        x = np.zeros(self.points_count) - 1
        y = np.zeros(self.points_count) - 1
        point_remote = [-1, -1]
        point_near_boundary = [-1, -1]
        rand_y = []
        rand_x = []
        poly, transform = None, None
        try_i = 0
        max_tries = 5
        while (
            not (
                # all points are inside image plane
                np.all(np.logical_and(x >= 0, x < inner_mask.shape[1]))
                and np.all(np.logical_and(y >= 0, y < inner_mask.shape[0]))
                # value scatter is limited to bbox size
                and max(x) - min(x) < bbox_w
                and max(y) - min(y) < bbox_h
                # most of the points are inside bbox
                and np.logical_and(x >= bbox[2], x <= bbox[3]).sum() > len(x) * 0.75
                and np.logical_and(y >= bbox[0], y <= bbox[1]).sum() > len(y) * 0.75
                # test for infinity and NaN
                and np.all(np.isfinite(x))
                and np.all(np.isfinite(y))
            )
            # avoid infinite loop
            and try_i < max_tries
        ):
            try_i += 1

            # Sample a point near the mask boundary
            boundary_size = get_boundary_size(inner_mask, dilation_ratio=0.01, use_mask_diag=False)
            boundary_vicinity_mask = mask_to_boundary(inner_mask.astype(np.uint8), boundary_size=boundary_size)
            boundary_vicinity = np.argwhere(boundary_vicinity_mask)
            if len(boundary_vicinity) == 0:
                boundary_vicinity = np.argwhere(inner_mask)
            point_near_boundary = boundary_vicinity[generator.randint(0, len(boundary_vicinity))]
            # Sample a point far enough from the boundary point
            # point_near_boundary <=> pnb
            pnb_distances = ((indices - point_near_boundary) ** 2).sum(axis=1)
            remote_positions = np.where(pnb_distances > np.quantile(pnb_distances[pnb_distances > 0], 0.7))[0]
            if len(remote_positions) > 0:
                pnb_remoteness = indices[remote_positions]
            else:
                pnb_remoteness = indices
            point_remote = pnb_remoteness[generator.randint(0, len(pnb_remoteness))]

            # Sample random points between two end points
            inbetween_bbox = (
                min(point_remote[0], point_near_boundary[0]),
                max(point_remote[0], point_near_boundary[0]),
                min(point_remote[1], point_near_boundary[1]),
                max(point_remote[1], point_near_boundary[1]),
            )
            inner_bbox_mask = (
                (indices[:, 0] > inbetween_bbox[0]) & (indices[:, 0] < inbetween_bbox[1])
                & (indices[:, 1] > inbetween_bbox[2]) & (indices[:, 1] < inbetween_bbox[3])
            )
            inbetween_indices = indices[inner_bbox_mask]
            if len(inbetween_indices) == 0:
                inbetween_indices = indices[
                    np.any(indices != point_remote, axis=1)
                    & np.any(indices != point_near_boundary, axis=1)
                ]

            polynom_degree = generator.randint(1, self.max_degree)
            fit_points_count = min(len(indices), generator.randint(2 * polynom_degree, 3 * polynom_degree + 1))
            rand_x = [point_remote[1], point_near_boundary[1]]
            rand_y = [point_remote[0], point_near_boundary[0]]
            fit_points_indices = generator.choice(
                range(len(inbetween_indices)),
                min(len(inbetween_indices), fit_points_count),
                replace=False
            )
            if len(fit_points_indices):
                rand_x = np.concatenate((rand_x, inbetween_indices[fit_points_indices, 1]))
                rand_y = np.concatenate((rand_y, inbetween_indices[fit_points_indices, 0]))

            bsize = bbox_w
            swap_xy = False
            if generator.rand() <= 0.5:
                bsize = bbox_h
                rand_y, rand_x = rand_x, rand_y
                swap_xy = True
            transform = self.axis_transform(bsize, min_period_ratio=0.3, generator=generator)

            y_std = max(np.std(rand_y), 1e-6)
            point_weights = np.ones_like(rand_x) / y_std
            # Force polynom to cover point_remote & point_near_boundary
            point_weights[:2] *= 2.
            x = np.linspace(min(rand_x), max(rand_x), self.points_count).astype(np.int32)
            with warnings.catch_warnings():
                # warning is raised when the rank of the coefficient matrix in the least-squares fit is deficient
                warnings.simplefilter('ignore', np.RankWarning)
                warnings.simplefilter('ignore', RuntimeWarning)
                warnings.simplefilter('ignore', UserWarning)
                try:
                    poly = Polynomial.fit(transform(rand_x), rand_y, polynom_degree, w=point_weights)
                    y = poly(transform(x)).astype(np.int32)
                except (SystemError, np.linalg.LinAlgError) as exc:
                    # logger.info(f'Ignore exception in StrokeGenerator: {str(exc)}')
                    continue
            if swap_xy:
                x, y = y, x
                rand_x, rand_y = rand_y, rand_x

        valid = False
        if np.all(np.isfinite(x)) & np.all(np.isfinite(y)):
            valid_coords_mask = (x > 0) & (y > 0)
            if valid_coords_mask.sum() > 0:
                valid_coords = np.stack([x, y], axis=1)[valid_coords_mask]
                valid_coords = [(y_c, x_c) for x_c, y_c in valid_coords]
                stroke = Stroke(is_positive, coords=valid_coords)
                valid = True
        if not valid:
            point_remote = [-1, -1]
            point_near_boundary = [-1, -1]
            rand_y = []
            rand_x = []

        meta['basis'] = rand_y, rand_x
        meta['ends'] = (point_remote[0], point_remote[1], point_near_boundary[0], point_near_boundary[1])
        meta['poly'] = poly
        meta['transform'] = transform
        if with_meta:
            return stroke, meta
        return stroke

    def generate_stroke_mask(self, mask, is_positive, with_meta=False):
        meta = None
        stroke = self.generate_stroke(mask > 0, is_positive=is_positive, with_meta=with_meta)
        if with_meta:
            stroke, meta = stroke
        strokes_mask = Stroke.get_mask(stroke, mask.shape, thickness=self.width, filled=False)
        strokes_mask = strokes_mask.astype(np.uint8)
        if with_meta:
            return strokes_mask, meta
        return strokes_mask
