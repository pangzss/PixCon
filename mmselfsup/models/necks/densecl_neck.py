# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn import build_norm_layer
from ..builder import NECKS


@NECKS.register_module()
class DenseCLNeck(BaseModule):
    """The non-linear neck of DenseCL.

    Single and dense neck in parallel: fc-relu-fc, conv-relu-conv.
    Borrowed from the authors' code: `<https://github.com/WXinlong/DenseCL`_.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_grid (int): The grid size of dense features. Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None,
                 init_cfg=None,
                 norm_cfg=dict(type='SyncBN'),
                 with_last_bias=True,
                 syncbn=False,
                 ):
        super(DenseCLNeck, self).__init__(init_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), 
            build_norm_layer(norm_cfg, hid_channels)[1] if syncbn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels, bias=with_last_bias))

        self.with_pool = True if num_grid is not None else False
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1),
            build_norm_layer(norm_cfg, hid_channels)[1] if syncbn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1, bias=with_last_bias))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, mode, weights=None):
        """Forward function of neck.

        Args:
            x (list[tensor]): feature map of backbone.
        """
        assert len(x) == 1
        assert mode in ["dense", "image"]
        x = x[0]
        if mode == "image":
            avgpooled_x = self.avgpool(x)
            avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))
            if weights is not None:
                wpooled_x = (x.view(x.shape[0], x.shape[1], -1) * weights.unsqueeze(1)).sum(-1)
                wpooled_x = wpooled_x / weights.sum(-1, keepdim=True)
                wpooled_x = self.mlp(wpooled_x) 
                return wpooled_x, avgpooled_x
            return avgpooled_x, avgpooled_x
        else: 
            if self.with_pool:
                x = self.pool(x)  # sxs
            x = self.mlp2(x)  # sxs: bxdxsxs
            avgpooled_x2 = self.avgpool2(x)  # 1x1: bxdx1x1
            x = x.view(x.size(0), x.size(1), -1)  # bxdxs^2
            avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1)  # bxd
            return x, avgpooled_x2