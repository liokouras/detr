# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Type, Union
import re

import torch
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter, _ovewrite_named_param
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

from varuna import CutPoint


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool, varuna: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        if varuna:
            self.body = backbone
        else:
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors) 
        # xs is the activations/output of layer4 of the backbone model.
        # singleton dict: {"0", layer4 output tensor}
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 varuna: bool):
        if varuna:
            backbone = ResNetVaruna(name, [False, False, dilation])
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), 
                norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, varuna)

class ResNetVaruna(models.resnet.ResNet):
    def __init__(self, name, replace_stride_with_dilation, **kwargs):
        if name == 'wide101':
            weights = models.Wide_ResNet101_2_Weights.verify(models.Wide_ResNet101_2_Weights.DEFAULT)
            layers = [3, 4, 23, 3]
        elif name == 'resnet101':
            weights = models.ResNet101_Weights.verify(models.ResNet101_Weights.DEFAULT)
            layers = [3, 4, 23, 3]
        else:
            # default to ResNet50
            weights = models.ResNet50_Weights.verify(models.ResNet50_Weights.DEFAULT)
            layers = [3, 4, 6, 3]

        weight_dict = self.rename_weights(weights.get_state_dict(progress=True)) # TODO rename layers for RN-50 !

        super().__init__(
            models.resnet.Bottleneck, layers,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=FrozenBatchNorm2d,
            width_per_group = (64 * 2) if name == 'wide101' else 64,
            num_classes = len(weights.meta["categories"]))

        self.load_state_dict(weight_dict)
        self.cp2to3 = CutPoint()
        self.cp3to4 = CutPoint()

    def _make_layer(
        self,
        block: Type[Union[models.resnet.BasicBlock, models.resnet.Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                models.resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # HACK: Layer1 is only layer with stride = 1 without dilation
            # Layer1 mustn't have CPs as it is not trainable.
            if stride > 1 or dilate:
                layers.append(CutPoint()) 
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        returndict = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.cp2to3(x)
        x = self.layer3(x)
        x = self.cp3to4(x)
        x = self.layer4(x)

        # make output compatible with what BackboneBase expects
        returndict["0"] = x

        return returndict

    def rename_weights(self, weights):
        # regular expression for ResNet layers containing cutpoints
        pattern = r"(layer(?!1).\.)(\d+)(\..*)"

        new_state_dict = {}

        for key, value in weights.items():
            match = re.match(pattern, key)

            if match:
                # block-index is doubled due to cutpoint insertion
                new_idx = match.group(1) + str(int(match.group(2)) * 2) + match.group(3)
                new_key = re.sub(pattern, new_idx, key)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        return new_state_dict


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.varuna)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
