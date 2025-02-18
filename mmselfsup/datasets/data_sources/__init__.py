# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseDataSource
from .cifar import CIFAR10, CIFAR100
from .image_list import ImageList
from .imagenet import ImageNet
from .imagenet_21k import ImageNet21k
from .coco import COCO
from .cocostuff27_source import COCOStuff27Source
# from .multi_mask_coco_source import MultiMaskCOCOSource
__all__ = [
    'BaseDataSource', 'CIFAR10', 'CIFAR100', 'ImageList', 'ImageNet',
    'ImageNet21k', 'COCO', 'COCOStuff27Source'
]
