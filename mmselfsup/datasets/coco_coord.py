# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import numpy as np
import glob
from PIL import Image
from mmcv.utils import build_from_cfg
from albumentations import Compose

from .base import BaseDataset
from .builder import DATASETS, ALBPIPELINES

@DATASETS.register_module()
class COCOCoord(BaseDataset):
    def __init__(self, dataset, main_dir, split, two_crop, hflip, pipelines, mask_size=7, return_coords=True, coco_plus_dir=None):
        if dataset == "COCO":
            main_dir = os.path.join(main_dir, split)
            self.all_imgs = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(main_dir)) for f in fn]
            if coco_plus_dir is not None:
                self.all_imgs += [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(coco_plus_dir)) for f in fn] 
        elif dataset == "ImageNet":
            self.all_imgs = list(glob.glob(main_dir + '/train/data/*/*.JPEG'))

        self.all_imgs = np.array(self.all_imgs)

        self.two_crop = build_from_cfg(two_crop, ALBPIPELINES)
        self.hflip = build_from_cfg(hflip, ALBPIPELINES) 

        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, ALBPIPELINES) for p in pipe])
            self.pipelines.append(pipeline)
        
        self.mask_size = mask_size
        self.return_coords = return_coords
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, idx):
        img_loc = self.all_imgs[idx]
        image = Image.open(img_loc).convert('RGB')
        h, w = image.size[:2]
        # Crop and flip
        crops, coords, coords_clipped = self.two_crop(image)
        crops, coords, coords_clipped, flags = self.hflip(crops, coords, coords_clipped)
        coords_rescaled = [(item * self.mask_size).long() for item in coords_clipped]
        masks = []
        for i in range(2):
            m = torch.zeros(self.mask_size, self.mask_size).long()
            m[coords_rescaled[i][1]:coords_rescaled[i][3],coords_rescaled[i][0]:coords_rescaled[i][2]] = 1
            masks.append(m)
        # augment
        crops_trans = []
        for i, crop in enumerate(crops):
            crop = self.pipelines[i](image=np.array(crop))["image"]
            crops_trans.append(crop)
        # query and key crops
        if self.return_coords:
            return dict(img=crops_trans,
                        coords=coords,
                        coords_clipped=coords_clipped,
                        flags=flags,
                        masks=masks
                        )
        else:  
            return dict(img=crops_trans,
                        masks=masks
                        )
                    

    def evaluate(self, results, logger=None):
        return NotImplemented