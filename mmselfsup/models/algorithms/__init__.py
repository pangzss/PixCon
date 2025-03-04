# Copyright (c) OpenMMLab. All rights reserved.
from .barlowtwins import BarlowTwins
from .base import BaseModel
from .byol import BYOL
from .cae import CAE
from .classification import Classification
from .deepcluster import DeepCluster
from .densecl import DenseCL
from .mae import MAE
from .maskfeat import MaskFeat
from .mmcls_classifier_wrapper import MMClsImageClassifierWrapper
from .moco import MoCo
from .mocov3 import MoCoV3
from .npid import NPID
from .odc import ODC
from .relative_loc import RelativeLoc
from .rotation_pred import RotationPred
from .simclr import SimCLR
from .simmim import SimMIM
from .simsiam import SimSiam
from .mocov2_plus import MoCoV2plus
from .pixcon import PixCon
__all__ = [
    'BaseModel', 'BarlowTwins', 'BYOL', 'Classification', 'DeepCluster',
    'DenseCL', 'MoCo', 'NPID', 'ODC', 'RelativeLoc', 'RotationPred', 'SimCLR',
    'SimSiam', 'SwAV', 'MAE', 'MoCoV3', 'SimMIM',
    'MMClsImageClassifierWrapper', 'CAE', 'MaskFeat',
    "MoCoV2plus", "PixCon"
]
