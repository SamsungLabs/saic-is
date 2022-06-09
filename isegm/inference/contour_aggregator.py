from copy import deepcopy

import numpy as np

from isegm.data.contour import Contour
from isegm.data.interaction_generators import ContourGenerator
from isegm.data.interaction_type import IType
from isegm.data.next_input_getters import get_contour_generation_params


class ContourAggregator(object):
    def __init__(
        self, gt_mask=None, init_contours=None,
        ignore_label=-1, contour_indx_offset=0, min_contour_size=10, augmentations=True,
        generator=ContourGenerator(deterministic=True, one_component=True),
    ):
        self.contour_indx_offset = contour_indx_offset
        self.min_contour_size = min_contour_size
        self.augmentations = augmentations
        self.generator = generator
        self.input_type = IType.contour
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset()

        if init_contours is not None:
            self.add_contours(init_contours)

    def make_next(self, pred_mask, **kwargs):
        assert self.gt_mask is not None
        contours = self._get_next_contours(pred_mask)
        self.add_contours(contours)

    def get_current(self):
        return self.current_contour

    def get_interactions(self, contours_limit=None):
        ilist = self.interactions_list[:contours_limit]
        if self.current_contour is not None:
            ilist.append([self.get_current()])
        return [sum(ilist, [])]

    def get_last_contours(self):
        return self.interactions_list[-1] if len(self.interactions_list) else None

    def get_flatten_contours(self):
        contours = [c for contours in self.interactions_list for c in contours]
        return contours

    def _get_next_contours(self, pred_mask):
        if pred_mask.sum() == 0:
            diff_map = self.gt_mask
            is_positive = True
        else:
            fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
            fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)
            diff_map, is_positive = get_contour_generation_params(
                self.gt_mask, fn_mask.astype(np.uint8), fp_mask.astype(np.uint8),
                dt_size=0, inner_mask_coeff=0.8
            )
        if diff_map is None:
            diff_map = self.gt_mask
        contour = self.generator.generate_contour(diff_map.copy(), is_positive=is_positive)
        return contour

    def begin(self, is_positive, coords=None):
        self.current_contour = Contour(is_positive, coords)

    def add_point(self, coords):
        self.current_contour.add_point(coords)
        if self.gt_mask is not None:
            self.not_covered_map[coords[0], coords[1]] = False

    def finish(self):
        if self.current_contour is not None:
            self.add_contours([self.current_contour])
        self.current_contour = None

    def add_contours(self, contours):
        if not isinstance(contours, list):
            contours = [contours]
        self.interactions_list.append(contours)
        for contour in contours:
            if contour.is_positive:
                self.num_pos_contours += 1
            else:
                self.num_neg_contours += 1

    def _remove_last_contours(self):
        contours = self.interactions_list.pop()

        for contour in contours:
            coords = contour.coords
            if contour.is_positive:
                self.num_pos_contours -= 1
            else:
                self.num_neg_contours -= 1

            if self.gt_mask is not None:
                for c in coords:
                    self.not_covered_map[c[0], c[1]] = True

    def reset(self):
        if self.gt_mask is not None:
            self.not_covered_map = np.ones_like(self.gt_mask, dtype=np.bool)

        self.num_pos_contours = 0
        self.num_neg_contours = 0

        self.interactions_list = []
        self.current_contour = None

    def get_state(self):
        return deepcopy(self.interactions_list)

    def set_state(self, state):
        self.reset()
        for contour in state:
            self.add_contours(contour)

    def set_index_offset(self, index_offset):
        self.contour_indx_offset = index_offset

    def __len__(self):
        return len(self.interactions_list)
