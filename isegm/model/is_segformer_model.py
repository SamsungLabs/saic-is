import torch.nn as nn

from isegm.model.is_model import ISModel
from isegm.model.modeling.mix_transformer import InteractionAsImageEmbed
from isegm.model.modeling.segformer import SegFormerNet
from isegm.model.modifiers import LRMult
from isegm.utils.serialization import serialize


class SegFormerModel(ISModel):
    @serialize
    def __init__(
        self, backbone_type='mit_b1', img_size=224, backbone_lr_mult=0.1,
        norm_layer=nn.BatchNorm2d, align_corners=False,
        **kwargs
    ):
        super().__init__(norm_layer=norm_layer, **kwargs)
        self.feature_extractor = SegFormerNet(
            num_classes=1, backbone_type=backbone_type,
            img_size=img_size, align_corners=align_corners,
            norm_cfg=dict(type='BN2d', requires_grad=True),
            with_aux_output=self.with_aux_output,
            with_prev_mask=self.with_prev_mask,
        )

        if self.maps_transform is not None:
            maps_embed_dims = [self.feature_extractor.backbone.embed_dims[0]]
            self.click_features_embed = InteractionAsImageEmbed(
                img_size=img_size, in_chans=self.maps_transform[-2].out_channels,
                embed_dims=maps_embed_dims
            )

        self.feature_extractor.apply(LRMult(backbone_lr_mult))

    def backbone_forward(self, image, coord_features=None):
        if self.maps_transform is not None:
            coord_features = self.click_features_embed(coord_features)
        net_outputs = self.feature_extractor(image, coord_features=coord_features)

        return {'instances': net_outputs[0], 'instances_aux': net_outputs[1]}

    def load_pretrained_weights(self, pretrained_path=''):
        import torch
        model_dict = self.state_dict()
        pretrained_dict = torch.load(pretrained_path, map_location={'cuda:0': 'cpu'})
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
