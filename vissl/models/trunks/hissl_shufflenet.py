# coding=utf-8
# Copyright (c) HISSL Contributors

import logging
from enum import Enum
from typing import List
from vissl.config import AttrDict
import torchvision.models as models

from vissl.models.model_helpers import (
    get_trunk_forward_outputs,
    transform_model_input_data_type,
)


import torch
import torch.nn as nn
from vissl.models.trunks import register_model_trunk
import torchvision.models.shufflenetv2 as shufflenetv2

# For different shufflenets, add the config here
WIDTH_CONFIG = {
    "v2x0.5": shufflenetv2.shufflenet_v2_x0_5,
    "v2x1.0": shufflenetv2.shufflenet_v2_x1_0,
    "v2x1.5": shufflenetv2.shufflenet_v2_x1_5,
    "v2x2.0": shufflenetv2.shufflenet_v2_x2_0,
}


class ShufflenetGlobalpool(nn.Module):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return globalpool as implemented in official torch shufflenet:
        https://pytorch.org/vision/0.10/_modules/torchvision/models/shufflenetv2.html
        """
        return x.mean([2, 3])


@register_model_trunk("shufflenet")
class HisslShuffleNet(nn.Module):
    """
    Implements the standard TorchVision ShufflenetV2 model.
    Currently, this does not have as many technical features as the models originally
    implemented by VISSL, like activation checkpointing.

    Required config
        config.MODEL.TRUNK.NAME=shufflenet
        config.MODEL.TRUNK.SHUFFLENET.WIDTH
            values: v2x0.5, v2x1.0, v2x1.5, v2x2.0
        config.MODEL.TRUNK.SHUFFLENET.PRETRAINED
            values: True, False
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(HisslShuffleNet, self).__init__()
        self.model_config = model_config

        logging.info("ShuffleNet trunk, does not yet support activation checkpointing")

        try:
            pretrained = model_config.TRUNK.SHUFFLENET.PRETRAINED
        except AttributeError:
            pretrained = False

        # get the params trunk takes from the config
        trunk_config = self.model_config.TRUNK.SHUFFLENET

        logging.info(f"Building model: ShuffleNet: {trunk_config.WIDTH}")
        try:
            self.model = WIDTH_CONFIG[trunk_config.WIDTH](pretrained=pretrained)
        except KeyError:
            logging.error(
                f"ShuffleNet config not found. User asked for {trunk_config.width}."
                f"Choose a value from {[key for key in WIDTH_CONFIG.keys()]}"
            )
            raise KeyError

        # implement the model trunk and construct all the layers that the trunk uses
        model_layer1 = self.model.conv1
        model_layer2 = self.model.maxpool
        model_layer3 = self.model.stage2
        model_layer4 = self.model.stage3
        model_layer5 = self.model.stage4
        model_layer6 = self.model.conv5
        model_layer7 = ShufflenetGlobalpool()

        # these features can be used for other purposes: like
        # feature extraction etc.
        self._feature_blocks = nn.ModuleDict(
            [
                ("conv1", model_layer1),
                ("maxpool", model_layer2),
                ("stage2", model_layer3),
                ("stage3", model_layer4),
                ("stage4", model_layer5),
                ("conv5", model_layer6),
                ("globalpool", model_layer7),
            ]
        )

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = []
    ) -> List[torch.Tensor]:
        # See the forward pass of resnext.py for reference of how additional features
        # can be implemented. For now, we do not require these advanced features.

        output = []

        # TODO implement more advanced features. See vissl's resnext implementation
        if len(out_feat_keys) > 0:
            raise NotImplementedError

        for i, (feature_name, feature_block) in enumerate(self._feature_blocks.items()):
            x = feature_block(x)

        # VISSL expects a list. It either contains one vector (the output), or
        # a list of requested intermediate features
        # For now, we only implement the output of the model.
        output.append(x)

        return output
