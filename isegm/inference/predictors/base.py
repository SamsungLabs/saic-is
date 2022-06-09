import torch
import torch.nn.functional as F
from torchvision import transforms

from isegm.data.contour import Contour
from isegm.data.interaction_type import IType
from isegm.data.stroke import Stroke
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, ResizeToFixedSize, LimitLongestSide


class BasePredictor(object):
    def __init__(
        self, model, device,
        net_interactions_limit=None,
        with_flip=False, zoom_in=None,
        max_size=None, fixed_size=None,
        **kwargs
    ):
        self.with_flip = with_flip
        self.net_interactions_limit = net_interactions_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.models_dict = None
        self.net = None

        if isinstance(model, dict):
            self.models_dict = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        if fixed_size is not None:
            self.transforms.append(ResizeToFixedSize(to_size=fixed_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def get_prediction(self, interactive_imitator, prev_mask=None, input_type=None):
        interactive_infos_dict = interactive_imitator.get_interactions()
        if self.models_dict:
            self.net = self.models_dict[input_type]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, interactive_infos_dicts, is_image_changed = self.apply_transforms(
            input_image, interactive_infos_dict, input_type
        )

        pred_logits = self._get_prediction(
            image_nd, interactive_infos_dicts, input_type,
            is_image_changed,
        )
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation(input_type):
            return self.get_prediction(interactive_imitator)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, interactive_infos_dicts, forced_input_type, is_image_changed):
        interactive_info_dict = {}
        for input_type, interactive_infos_lists in interactive_infos_dicts.items():
            if input_type == IType.point:
                interactive_info = self.get_points_nd(interactive_infos_lists, image_nd.shape)
            elif input_type == IType.stroke:
                interactive_info = self.get_strokes_nd(interactive_infos_lists, image_nd.shape, width=10)
            elif input_type == IType.contour:
                interactive_info = self.get_contours_nd(interactive_infos_lists, image_nd.shape, width=10)
            else:
                raise NotImplementedError
            interactive_info_dict[f'{input_type.name}_interactive_info'] = interactive_info
        return self.net(
            image_nd, interactive_info_dict,
            input_type=forced_input_type,
        )['instances']

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, interactive_infos_dict, input_type):
        is_image_changed = False
        for t in self.transforms:
            image_nd, interactive_infos_dict = t.transform(image_nd, interactive_infos_dict, input_type)
            is_image_changed |= t.image_changed

        return image_nd, interactive_infos_dict, is_image_changed

    def get_points_nd(self, clicks_lists, image_shape):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_interactions_limit is not None:
            num_max_points = min(self.net_interactions_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_interactions_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)
        return torch.tensor(total_clicks, device=self.device)

    def get_strokes_nd(self, strokes_lists, image_shape, width):
        strokes_mask = torch.zeros((image_shape[0], 2, image_shape[2], image_shape[3]),
                                   dtype=torch.float32, device=self.device)
        for idx, strokes in enumerate(strokes_lists):
            mask_i = Stroke.get_mask(strokes, image_shape[-2:], thickness=width, filled=False)
            mask_i = torch.from_numpy(mask_i).to(self.device)
            strokes_mask[idx, :, :, :] = mask_i
        return strokes_mask

    def get_contours_nd(self, contours_lists, image_shape, width):
        contours_mask = torch.zeros((image_shape[0], 2, image_shape[2], image_shape[3]),
                                    dtype=torch.float32, device=self.device)
        for idx, contours in enumerate(contours_lists):
            mask_i = Contour.get_mask(contours, image_shape[-2:], filled=True, thickness=width)
            mask_i = torch.from_numpy(mask_i).to(self.device)
            contours_mask[idx, :, :, :] = mask_i

        return contours_mask

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
