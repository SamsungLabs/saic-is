from typing import Iterable
import torch
import torch.nn as nn

from isegm.data.interaction_type import IType
from isegm.model.ops import DistMaps, ContoursMaps, StrokesMaps, ScaleLayer, BatchImageNormalize


class ISModel(nn.Module):
    def __init__(
        self, use_rgb_conv=True, with_aux_output=False,
        norm_radius=260, use_disks=False,
        with_prev_mask=False, use_leaky_relu=False,
        binary_prev_mask=False, norm_layer=nn.BatchNorm2d,
        norm_mean_std=([.485, .456, .406], [.229, .224, .225]),
        is_strided_maps_transform=False,
        is_training=False,
        input_type=IType.point,
        accumulate_multi_input=False
    ):
        super().__init__()
        self.input_type = input_type
        self.multi_input = isinstance(self.input_type, Iterable)
        if self.multi_input:
            self.itype_map = {itype: i for i, itype in enumerate(self.input_type)}
        self.accumulate_multi_input = accumulate_multi_input
        self.is_training = is_training
        self.with_aux_output = with_aux_output
        self.clicks_groups = None
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.coord_feature_ch = 2 * len(self.input_type) if self.multi_input else 2

        if self.with_prev_mask:
            self.coord_feature_ch += 1

        self.maps_transform = None
        if use_rgb_conv:
            rgb_conv_layers = [
                nn.Conv2d(in_channels=3 + self.coord_feature_ch, out_channels=6 + self.coord_feature_ch, kernel_size=1),
                norm_layer(6 + self.coord_feature_ch),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6 + self.coord_feature_ch, out_channels=3, kernel_size=1)
            ]
            self.rgb_conv = nn.Sequential(*rgb_conv_layers)
        else:
            self.rgb_conv = None
            mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=16, out_channels=64,
                    kernel_size=3, padding=1,
                    stride=1 if not is_strided_maps_transform else 2
                ),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            self.maps_transform = nn.Sequential(*mt_layers)

        if self.multi_input:
            self.dist_maps = nn.ModuleDict({
                k.name: v for k, v in (
                    (IType.point, DistMaps(norm_radius=norm_radius, spatial_scale=1.0, use_disks=use_disks)),
                    (IType.contour, ContoursMaps()),
                    (IType.stroke, StrokesMaps())
                )
                if k in self.input_type
            })
        else:
            if self.input_type.name == IType.point.name:
                self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0, use_disks=use_disks)
            elif self.input_type.name == IType.contour.name:
                self.dist_maps = ContoursMaps()
            elif self.input_type.name == IType.stroke.name:
                self.dist_maps = StrokesMaps()
            else:
                raise NotImplementedError

    def forward(self, image, interactive_info, input_type):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(
            image, prev_mask,
            interactive_info=interactive_info, input_type=input_type,
        )

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
            outputs = self.backbone_forward(x)
        else:
            coord_features = self.maps_transform(coord_features)
            outputs = self.backbone_forward(image, coord_features)

        outputs['instances'] = nn.functional.interpolate(
            outputs['instances'], size=image.size()[2:], mode='bilinear', align_corners=True
        )
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(
                outputs['instances_aux'], size=image.size()[2:], mode='bilinear', align_corners=True
            )

        return outputs

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()

        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, interactive_info, input_type):
        if self.multi_input:
            if self.accumulate_multi_input:
                coord_features = [
                    self.dist_maps[itype.name](
                        image, interactive_info[f'{itype.name}_interactive_info'],
                    ) for itype in self.input_type
                ]
                coord_features = torch.cat(coord_features, dim=1)
            else:
                assert input_type.name in self.dist_maps
                coord_features = torch.zeros(
                    (image.shape[0], len(self.input_type) * 2, *image.shape[2:]),
                    device=image.device
                )
                coord_features_itype = self.dist_maps[input_type.name](
                    image, interactive_info[f'{input_type.name}_interactive_info'],
                )
                itype_i = self.itype_map[input_type] * 2
                coord_features[:, itype_i:itype_i + 2] = coord_features_itype
        else:
            coord_features = self.dist_maps(image, interactive_info[f'{input_type.name}_interactive_info'])

        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features
