from copy import deepcopy

import numpy as np

from isegm.data.interaction_generators import PointGenerator
from isegm.data.interaction_type import IType
from isegm.inference.utils import get_fn_fp_distance_transform


class Clicker(object):
    def __init__(
        self,
        gt_mask=None, init_clicks=None,
        ignore_label=-1, click_indx_offset=0,
        generator=PointGenerator(at_max_mask=True),
    ):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None
        self.generator = generator
        self.input_type = IType.point

        self.reset()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_point(click)

    def make_next(self, pred_mask, **kwargs):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        if click is not None:
            self.add_point(click)

    def get_interactions(self, clicks_limit=None):
        return self.interactions_list[:clicks_limit]

    def _get_next_click(self, pred_mask, padding=True):
        fn_mask, fn_mask_dt, fp_mask, fp_mask_dt = get_fn_fp_distance_transform(
            pred_mask, self.gt_mask,
            self.not_clicked_map, self.not_ignore_mask, padding
        )
        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)
        is_positive = fn_max_dist > fp_max_dist
        mask = fn_mask if is_positive else fp_mask
        coords = self.generator.generate_points(mask, is_positive=is_positive, num_points=1)
        if len(coords) == 0:
            return None
        coords_y, coords_x = coords[0][:2]

        return Click(is_positive=is_positive, coords=(coords_y, coords_x))

    def add_point(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.interactions_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.interactions_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.interactions_list = []

    def get_state(self):
        return deepcopy(self.interactions_list)

    def set_state(self, state):
        self.reset()
        for click in state:
            self.add_point(click)

    def set_index_offset(self, index_offset):
        self.click_indx_offset = index_offset

    def __len__(self):
        return len(self.interactions_list)


class Click:
    def __init__(self, is_positive, coords, indx=None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        return *self.coords, self.indx

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy
