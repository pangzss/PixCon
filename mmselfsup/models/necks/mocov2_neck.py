# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn import build_norm_layer

from ..builder import NECKS


@NECKS.register_module()
class MoCoV2Neck(BaseModule):
    """The non-linear neck of MoCo v2: fc-relu-fc.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 norm_cfg=dict(type='SyncBN'),
                 init_cfg=None,
                 syncbn=False,
                 with_last_bias=True):
        super(MoCoV2Neck, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            build_norm_layer(norm_cfg, hid_channels)[1] if syncbn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels, bias=with_last_bias))

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]