from isegm.data.interaction_type import IType
from .zoom_in import ZoomIn, get_roi_image_nd


class LimitLongestSide(ZoomIn):
    def __init__(self, max_size=800):
        super().__init__(target_size=max_size, skip_interactions=0)

    def transform(self, image_nd, interaction_dict, input_type):
        assert image_nd.shape[0] == 1
        image_max_size = max(image_nd.shape[2:4])
        self.image_changed = False

        if image_max_size <= self.target_size:
            return image_nd, interaction_dict
        self._input_image = image_nd

        self._object_roi = (0, image_nd.shape[2] - 1, 0, image_nd.shape[3] - 1)
        self._roi_image = get_roi_image_nd(image_nd, self._object_roi, self.target_size)
        self.image_changed = True

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
        return self._roi_image, ti_dict
