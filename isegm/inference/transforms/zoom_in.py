from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F

from isegm.data.interaction_type import IType
from isegm.data.stroke import Stroke
from isegm.inference.contour_aggregator import Contour
from isegm.utils.misc import get_bbox_iou, get_bbox_from_mask, expand_bbox, clamp_bbox
from .base import BaseTransform


class ZoomIn(BaseTransform):
    def __init__(
        self,
        target_size=400,
        skip_interactions=1,
        expansion_ratio=1.4, min_crop_size=200,
        recompute_thresh_iou=0.5, prob_thresh=0.50,
    ):
        super().__init__()
        self.target_size = target_size
        self.min_crop_size = min_crop_size
        if not isinstance(skip_interactions, dict):
            self.skip_interactions = {ui_type: skip_interactions for ui_type in IType}
        else:
            self.skip_interactions = skip_interactions
        self.expansion_ratio = expansion_ratio
        self.recompute_thresh_iou = recompute_thresh_iou
        self.prob_thresh = prob_thresh

        self._input_image_shape = None
        self._prev_probs = None
        self._object_roi = None
        self._roi_image = None
        self._interaction_i = 0
        self._input_type = None

    def transform(self, image_nd, interaction_dict: Dict[IType, List], input_type):
        assert image_nd.shape[0] == 1
        self.image_changed = False
        self._interaction_i += 1
        self._input_type = input_type
        skip_interactions = self.skip_interactions[self._input_type]

        if self._interaction_i <= skip_interactions:
            return image_nd, interaction_dict

        self._input_image_shape = image_nd.shape

        if self._prev_probs is not None:
            current_pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
        else:
            current_pred_mask = np.zeros(image_nd.shape[-2:])
        current_object_roi = get_object_roi(current_pred_mask, interaction_dict,
                                            self.expansion_ratio, self.min_crop_size)

        update_object_roi = (
            self._object_roi is None
            or not is_interaction_inside_object(self._object_roi, interaction_dict)
            or get_bbox_iou(current_object_roi, self._object_roi) < self.recompute_thresh_iou
        )
        if update_object_roi:
            self._object_roi = current_object_roi
            self.image_changed = True
        self._roi_image = get_roi_image_nd(image_nd, self._object_roi, self.target_size)

        ti_dict = {}
        for input_type, interaction_list in interaction_dict.items():
            if input_type == IType.point:
                ti_lists = self._transform_clicks(interaction_list)
            elif input_type == IType.stroke:
                ti_lists = self._transform_strokes(interaction_list)
            elif input_type == IType.contour:
                ti_lists = self._transform_contours(interaction_list)
            else:
                raise NotImplementedError
            ti_dict[input_type] = ti_lists
        return self._roi_image.to(image_nd.device), ti_dict

    def inv_transform(self, prob_map):
        if self._object_roi is None:
            self._prev_probs = prob_map.cpu().numpy()
            return prob_map

        assert prob_map.shape[0] == 1
        rmin, rmax, cmin, cmax = self._object_roi
        prob_map = torch.nn.functional.interpolate(prob_map, size=(rmax - rmin + 1, cmax - cmin + 1),
                                                   mode='bilinear', align_corners=True)

        skip_interactions = self.skip_interactions[self._input_type]
        if self._prev_probs is not None:
            new_prob_map = torch.zeros(*self._prev_probs.shape, device=prob_map.device, dtype=prob_map.dtype)
            new_prob_map[:, :, rmin:rmax + 1, cmin:cmax + 1] = prob_map
        elif self._interaction_i > skip_interactions:
            shape = list(self._input_image_shape)
            shape[1] = 1
            new_prob_map = torch.zeros(*shape, device=prob_map.device, dtype=prob_map.dtype)
            new_prob_map[:, :, rmin:rmax + 1, cmin:cmax + 1] = prob_map
        else:
            new_prob_map = prob_map

        self._prev_probs = new_prob_map.cpu().numpy()

        return new_prob_map

    def check_possible_recalculation(self, input_type):
        if (
            self._prev_probs is None
            or self._object_roi is not None
            or self.skip_interactions[input_type] > 0
            or self._interaction_i > self.skip_interactions[input_type]
        ):
            return False

        pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
        if pred_mask.sum() > 0:
            possible_object_roi = get_object_roi(pred_mask, [],
                                                 self.expansion_ratio, self.min_crop_size)
            image_roi = (0, self._input_image_shape[2] - 1, 0, self._input_image_shape[3] - 1)
            if get_bbox_iou(possible_object_roi, image_roi) < 0.50:
                return True
        return False

    def get_state(self):
        roi_image = self._roi_image.cpu() if self._roi_image is not None else None
        return self._input_image_shape, self._object_roi, self._prev_probs, roi_image, self.image_changed

    def set_state(self, state):
        self._input_image_shape, self._object_roi, self._prev_probs, self._roi_image, self.image_changed = state

    def reset(self):
        self._input_image_shape = None
        self._object_roi = None
        self._prev_probs = None
        self._roi_image = None
        self.image_changed = False
        self._interaction_i = 0
        self._input_type = None

    def _transform_clicks(self, clicks_list):
        if self._object_roi is None:
            return clicks_list

        rmin, rmax, cmin, cmax = self._object_roi
        crop_height, crop_width = self._roi_image.shape[2:]

        transformed_clicks = []
        for click in clicks_list:
            new_r = crop_height * (click.coords[0] - rmin) / (rmax - rmin + 1)
            new_c = crop_width * (click.coords[1] - cmin) / (cmax - cmin + 1)
            transformed_clicks.append(click.copy(coords=(new_r, new_c)))
        return transformed_clicks

    def _transform_strokes(self, strokes_list):
        if self._object_roi is None:
            return strokes_list
        rmin, rmax, cmin, cmax = self._object_roi
        crop_height, crop_width = self._roi_image.shape[2:]
        transformed_strokes = []
        for stroke in strokes_list:
            if len(stroke) == 0:
                transformed_strokes.append(stroke)
                continue
            coords = np.array(stroke.coords)
            coords[:, 0] = crop_height / (rmax - rmin + 1) * (coords[:, 0] - rmin)
            coords[:, 1] = crop_width / (cmax - cmin + 1) * (coords[:, 1] - cmin)
            t_stroke = Stroke(is_positive=stroke.is_positive, coords=[(y, x) for y, x in coords])
            transformed_strokes.append(t_stroke)
        return transformed_strokes

    def _transform_contours(self, contours_list):
        if self._object_roi is None:
            return contours_list
        rmin, rmax, cmin, cmax = self._object_roi
        crop_height, crop_width = self._roi_image.shape[2:]
        transformed_contours = []
        for contours in contours_list:
            t_contours = []
            for contour in contours:
                if len(contour) == 0:
                    t_contours.append(contour)
                    continue
                coords = np.array(contour.coords)
                coords[:, 0] = crop_height / (rmax - rmin + 1) * (coords[:, 0] - rmin)
                coords[:, 1] = crop_width / (cmax - cmin + 1) * (coords[:, 1] - cmin)
                t_contour = Contour(is_positive=contour.is_positive, coords=[(y, x) for y, x in coords])
                t_contours.append(t_contour)
            transformed_contours.append(t_contours)
        return transformed_contours


def get_object_roi(pred_mask, interactions_dict, expansion_ratio, min_crop_size):
    pred_mask = pred_mask.copy()
    for input_type, interactions_list in interactions_dict.items():
        if input_type == IType.point:
            for click in interactions_list:
                if click.is_positive:
                    pred_mask[int(click.coords[0]), int(click.coords[1])] = 1
        elif input_type == IType.contour:
            for contours in interactions_list:
                for contour in contours:
                    if contour.is_positive and len(contour):
                        yx = (np.array(contour.coords) + 0.5).astype(np.int32)
                        pred_mask[yx[:, 0], yx[:, 1]] = 1
        elif input_type == IType.stroke:
            for stroke in interactions_list:
                if stroke.is_positive and len(stroke):
                    yx = (np.array(stroke.coords) + 0.5).astype(np.int32)
                    yx_max = yx.max(axis=0)
                    yx_min = yx.min(axis=0)
                    yx_bbox = yx_max - yx_min
                    if yx_bbox[0] > yx_bbox[1]:
                        pred_mask[yx_min[0]:yx_max[0]] = 1
                    else:
                        pred_mask[:, yx_min[1]:yx_max[1]] = 1
        else:
            raise NotImplementedError

    bbox = get_bbox_from_mask(pred_mask)
    bbox = expand_bbox(bbox, expansion_ratio, min_crop_size)
    h, w = pred_mask.shape[0], pred_mask.shape[1]
    bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

    return bbox


def get_roi_image_nd(image_nd, object_roi, target_size):
    rmin, rmax, cmin, cmax = object_roi

    height = rmax - rmin + 1
    width = cmax - cmin + 1

    if isinstance(target_size, tuple):
        new_height, new_width = target_size
    else:
        scale = target_size / max(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))

    with torch.no_grad():
        roi_image_nd = image_nd[:, :, rmin:rmax + 1, cmin:cmax + 1]
        if (
            (isinstance(target_size, tuple) and target_size[0] > 0)
            or (not isinstance(target_size, tuple) and target_size > 0)
        ):
            roi_image_nd = torch.nn.functional.interpolate(roi_image_nd, size=(new_height, new_width),
                                                           mode='bilinear', align_corners=True)

    return roi_image_nd


def is_interaction_inside_object(object_roi, interactions_dict):
    for input_type, interactions_list in interactions_dict.items():
        for interaction in interactions_list:
            if input_type == IType.point:
                if interaction.is_positive:
                    if interaction.coords[0] < object_roi[0] or interaction.coords[0] >= object_roi[1]:
                        return False
                    if interaction.coords[1] < object_roi[2] or interaction.coords[1] >= object_roi[3]:
                        return False
            elif input_type == IType.stroke and len(interaction):
                coords = np.array(interaction.coords)
                if np.any(coords[:, 0] < object_roi[0]) or np.any(coords[:, 0] >= object_roi[1]):
                    return False
                if np.any(coords[:, 1] < object_roi[2]) or np.any(coords[:, 1] >= object_roi[3]):
                    return False
            elif input_type == IType.contour:
                for contour in interaction:
                    if len(contour):
                        coords = np.array(contour.coords)
                        if np.any(coords[:, 0] < object_roi[0]) or np.any(coords[:, 0] >= object_roi[1]):
                            return False
                        if np.any(coords[:, 1] < object_roi[2]) or np.any(coords[:, 1] >= object_roi[3]):
                            return False
    return True
