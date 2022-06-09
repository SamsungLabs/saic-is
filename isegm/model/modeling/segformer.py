import os

import torch
import torch.nn as nn
import torch._utils

from .mix_transformer import BACKBONES
from .segformer_head import SegFormerHead

DECODER_EMBED_DIMS = {
    'mit_b0': 256,
    'mit_b1': 256,
    'mit_b2': 768,
    'mit_b3': 768,
    'mit_b4': 768,
    'mit_b5': 768,
}


class SegFormerNet(nn.Module):
    def __init__(
        self, num_classes, backbone_type='mit_b1', img_size=(224, 224),
        align_corners=False, norm_cfg=dict(type='SyncBN', requires_grad=True),
        with_aux_output=False, with_prev_mask=False
    ):
        super(SegFormerNet, self).__init__()
        self.align_corners = align_corners
        self.num_classes = num_classes
        self.with_prev_mask = with_prev_mask

        self.backbone = BACKBONES[backbone_type](img_size=img_size, with_prev_mask=self.with_prev_mask)
        self.decode_head = SegFormerHead(
            feature_strides=[4, 8, 16, 32],
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=128,
            dropout_ratio=0.1,
            num_classes=self.num_classes,
            align_corners=self.align_corners,
            decoder_params=dict(embed_dim=DECODER_EMBED_DIMS[backbone_type]),
            norm_cfg=norm_cfg,
            with_aux_output=with_aux_output,
         )

    def forward(self, img, coord_features=None):
        x = self.backbone(img, coord_features=coord_features)
        out, out_aux = self.decode_head.forward(x, img_input=img)
        return [out, out_aux]

    def load_pretrained_weights(self, pretrained_path='', add_preffix=None):
        model_dict = self.state_dict()

        if not os.path.exists(pretrained_path):
            print(f'\nFile "{pretrained_path}" does not exist.')
            print('You need to specify the correct path to the pre-trained weights.\n'
                  'You can download the weights for HRNet from the repository:\n'
                  'https://github.com/HRNet/HRNet-Image-Classification')
            exit(1)
        pretrained_dict = torch.load(pretrained_path, map_location={'cuda:0': 'cpu'})
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        if add_preffix is not None:
            pretrained_dict = {add_preffix + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {
            k.replace('last_layer', 'aux_head').replace('model.', '').replace('feature_extractor.', ''): v
            for k, v in pretrained_dict.items()
        }

        for k in list(pretrained_dict.keys()):
            if k.startswith('decode_head.conv_seg'):
                del pretrained_dict[k]
            if k.startswith('decode_head.linear_pred'):
                del pretrained_dict[k]

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
