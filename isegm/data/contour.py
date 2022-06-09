from copy import deepcopy
import json
from typing import Iterable

import cv2
import numpy as np


class Contour:
    def __init__(self, is_positive, coords=None):
        self.is_positive = is_positive
        self.coords = coords if coords is not None else []

    def add_point(self, coord):
        self.coords.append(coord)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy

    @classmethod
    def get_mask(cls, contours, mask_shape, filled, thickness=10):
        if not isinstance(contours, Iterable):
            contours = [contours]
        if filled:
            mask = cls.get_filled_mask(contours, mask_shape)
        else:
            mask = cls.get_unfilled_mask(contours, mask_shape, thickness=thickness)
        mask = mask.transpose((2, 0, 1))
        return mask

    @classmethod
    def get_unfilled_mask(cls, contours, mask_shape, thickness=10):
        pos_mask = np.zeros(mask_shape, dtype=np.uint8)
        neg_mask = np.zeros(mask_shape, dtype=np.uint8)
        for contour in contours:
            if len(contour) > 2:
                pts = np.array(contour.coords)[:, ::-1].astype(np.int32)
                if contour.is_positive:
                    pos_mask = cv2.polylines(pos_mask, [pts], False, 1, thickness)
                else:
                    neg_mask = cv2.polylines(neg_mask, [pts], False, 1, thickness)
        mask = np.stack((pos_mask, neg_mask), axis=2)
        return mask

    @classmethod
    def get_filled_mask(cls, contours, mask_shape):
        pos_mask = np.zeros(mask_shape, dtype=np.uint8)
        neg_mask = np.zeros(mask_shape, dtype=np.uint8)
        for contour in contours:
            filled_mask = np.zeros(mask_shape, dtype=np.uint8)
            if len(contour) > 2:
                pts = [np.array(contour.coords)[:, ::-1].astype(np.int32)]
                cv2.fillPoly(filled_mask, pts=pts, color=1)
            if contour.is_positive:
                pos_mask = np.maximum(pos_mask, filled_mask)
            else:
                neg_mask = np.maximum(neg_mask, filled_mask)
        mask = np.stack((pos_mask, neg_mask), axis=2)
        return mask

    @classmethod
    def save(cls, file, contours):

        store_obj = []
        for contour in contours:
            coords = contour.coords
            is_pos = contour.is_positive
            obj = (is_pos, coords)
            store_obj.append(obj)

        with open(file, 'w') as fp:
            json.dump(store_obj, fp, indent=4)

    @classmethod
    def load(cls, file):
        with open(file, 'r') as fp:
            data = json.load(fp)
        contours = []
        for item in data:
            is_pos = item[0]
            coords = item[1]
            contours.append(Contour(is_pos, coords))
        return contours

    def is_empty(self):
        return len(self.coords) == 0

    def __len__(self):
        return len(self.coords)

    def __repr__(self):
        return (
            f'<{self.__class__.__module__}.{self.__class__.__name__}'
            f'(len={len(self)}, is_pos={self.is_positive}, coords={self.coords[:5]}) '
            f'object at {hex(id(self))}>'
        )
