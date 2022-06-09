from typing import List, Dict

import numpy as np
import torch

from isegm.data.interaction_type import IType
from isegm.data.stroke import Stroke
from isegm.inference.contour_aggregator import Contour
from .base import BaseTransform


class AddHorizontalFlip(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, image_nd, interaction_dict: Dict[IType, List], input_type):
        assert len(image_nd.shape) == 4
        image_nd = torch.cat([image_nd, torch.flip(image_nd, dims=[3])], dim=0)

        image_width = image_nd.shape[3]
        t_interaction_dict = {}
        for input_type, interaction_list in interaction_dict.items():
            if input_type == IType.point:
                interaction_lists_flipped = self._transform_clicks([interaction_list], image_width)
                interaction_lists = [interaction_list] + interaction_lists_flipped
            elif input_type == IType.stroke:
                interaction_lists_flipped = self._transform_strokes(interaction_list, image_width)
                interaction_lists = [interaction_list] + [interaction_lists_flipped]
            elif input_type == IType.contour:
                interaction_lists_flipped = self._transform_contours(interaction_list, image_width)
                interaction_lists = interaction_list + interaction_lists_flipped
            else:
                raise NotImplementedError
            t_interaction_dict[input_type] = interaction_lists
        return image_nd, t_interaction_dict

    def _transform_clicks(self, clicks_lists, image_width):
        clicks_lists_flipped = []
        for clicks_list in clicks_lists:
            clicks_list_flipped = [click.copy(coords=(click.coords[0], image_width - click.coords[1] - 1))
                                   for click in clicks_list]
            clicks_lists_flipped.append(clicks_list_flipped)
        return clicks_lists_flipped

    def _transform_strokes(self, strokes, image_width):
        strokes_flipped = []
        for stroke in strokes:
            if len(stroke) == 0:
                strokes_flipped.append(stroke)
                continue
            coords = np.array(stroke.coords)
            coords[:, 1] = image_width - coords[:, 1] - 1
            stroke_flipped = Stroke(is_positive=stroke.is_positive, coords=[(y, x) for y, x in coords])
            strokes_flipped.append(stroke_flipped)
        return strokes_flipped

    def _transform_contours(self, contours_lists, image_width):
        contours_lists_flipped = []
        for contours in contours_lists:
            contours_flipped = []
            for contour in contours:
                if len(contour) == 0:
                    contours_flipped.append(contour)
                    continue
                coords = np.array(contour.coords)
                coords[:, 1] = image_width - coords[:, 1] - 1
                contour_flipped = Contour(is_positive=contour.is_positive, coords=[(y, x) for y, x in coords])
                contours_flipped.append(contour_flipped)
            contours_lists_flipped.append(contours_flipped)
        return contours_lists_flipped

    def inv_transform(self, prob_map):
        assert len(prob_map.shape) == 4 and prob_map.shape[0] % 2 == 0
        num_maps = prob_map.shape[0] // 2
        prob_map, prob_map_flipped = prob_map[:num_maps], prob_map[num_maps:]
        return 0.5 * (prob_map + torch.flip(prob_map_flipped, dims=[3]))

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def reset(self):
        pass
