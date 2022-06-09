import numpy as np
import torch
from typing import List, Dict

from isegm.data.interaction_type import IType
from isegm.data.stroke import Stroke
from isegm.inference.contour_aggregator import Contour
from .base import BaseTransform


class ResizeToFixedSize(BaseTransform):
    def __init__(self, to_size=(320, 480)):
        super().__init__()
        self.to_height, self.to_width = to_size
        self._input_image_shape = None
        self._resized_image = None

    def transform(self, image_nd, interaction_dict: Dict[IType, List], input_type):
        assert image_nd.shape[0] == 1
        self._input_image_shape = image_nd.shape
        with torch.no_grad():
            self._resized_image = torch.nn.functional.interpolate(
                image_nd, size=(self.to_height, self.to_width),
                mode='bilinear', align_corners=True
            )

        t_interactions_dict = {}
        for input_type, interaction_list in interaction_dict.items():
            if input_type == IType.point:
                t_interactions = self._transform_clicks(interaction_list, image_nd.shape)
            elif input_type == IType.contour:
                t_interactions = self._transform_contours(interaction_list, image_nd.shape)
            elif input_type == IType.stroke:
                t_interactions = self._transform_strokes(interaction_list, image_nd.shape)
            else:
                raise NotImplementedError
            t_interactions_dict[input_type] = t_interactions

        return self._resized_image, t_interactions_dict

    def _transform_clicks(self, clicks_list, image_nd_shape):
        transformed_clicks = []
        for click in clicks_list:
            new_r = self.to_height * click.coords[0] / image_nd_shape[2]
            new_c = self.to_width * click.coords[1] / image_nd_shape[3]
            transformed_clicks.append(click.copy(coords=(new_r, new_c)))
        return transformed_clicks

    def _transform_strokes(self, strokes_lists, image_nd_shape):
        strokes_lists_n = []
        for stroke in strokes_lists:
            if len(stroke) == 0:
                strokes_lists_n.append(stroke)
                continue
            coords = np.array(stroke.coords)
            coords[:, 0] = self.to_height * coords[:, 0] / image_nd_shape[2]
            coords[:, 1] = self.to_width * coords[:, 1] / image_nd_shape[3]
            stroke_n = Stroke(is_positive=stroke.is_positive, coords=[(y, x) for y, x in coords])
            strokes_lists_n.append(stroke_n)
        return strokes_lists_n

    def _transform_contours(self, contours_lists, image_nd_shape):
        contours_lists_n = []
        for contours in contours_lists:
            contours_n = []
            for contour in contours:
                if len(contour) == 0:
                    contours_n.append(contour)
                    continue
                coords = np.array(contour.coords)
                coords[:, 0] = self.to_height * coords[:, 0] / image_nd_shape[2]
                coords[:, 1] = self.to_width * coords[:, 1] / image_nd_shape[3]
                contour_n = Contour(is_positive=contour.is_positive, coords=[(y, x) for y, x in coords])
                contours_n.append(contour_n)
            contours_lists_n.append(contours_n)
        return contours_lists_n

    def inv_transform(self, prob_map):
        assert prob_map.shape[0] == 1
        _, _, from_height, from_width = self._input_image_shape
        new_prob_map = torch.nn.functional.interpolate(
            prob_map, size=(from_height, from_width),
            mode='bilinear', align_corners=True
        )
        return new_prob_map

    def get_state(self):
        return self._input_image_shape, self._resized_image

    def set_state(self, state):
        self._input_image_shape, self._resized_image = state

    def reset(self):
        self._input_image_shape = None
        self._resized_image = None
