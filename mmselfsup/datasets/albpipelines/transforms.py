# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import random
import math
import warnings
import torch

import torch.nn as nn
import torchvision.transforms.functional as TF

import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2

from ..builder import ALBPIPELINES

# register all existing transforms in albumentations
_EXCLUDED_TRANSFORMS = ['ColorJitter', 'ToTensorV2']
for m in inspect.getmembers(alb, inspect.isclass):
    if m[0] not in _EXCLUDED_TRANSFORMS:
        ALBPIPELINES.register_module(m[1])

def _get_image_size(img):
    if TF._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

def _compute_intersection(box1, box2):
    i1, j1, h1, w1 = box1
    i2, j2, h2, w2 = box2
    x_overlap = max(0, min(j1+w1, j2+w2) - max(j1, j2))
    y_overlap = max(0, min(i1+h1, i2+h2) - max(i1, i2))
    return x_overlap * y_overlap

def _get_coord(i, j, h, w):
    coord = torch.Tensor([j, i, j + w, i + h])
    return coord

def _clip_coords(coords, params):
    # Two operations are performed in this function.
    # 1. extract intersecting area (x1_n,y1_n,x2_n,y2_n)
    # 2. shift the origin of intersecting area w.r.t the raw image to the origin of two crops, as
    # the intersecting area is defined w.r.t to original image. 
    # w.r.t to their own origins.
    x1_q, y1_q, x2_q, y2_q = coords[0]
    x1_k, y1_k, x2_k, y2_k = coords[1]
    _, _, height_q, width_q = params[0]
    _, _, height_k, width_k = params[1]

    x1_n, y1_n = torch.max(x1_q, x1_k), torch.max(y1_q, y1_k)
    x2_n, y2_n = torch.min(x2_q, x2_k), torch.min(y2_q, y2_k)
    
    coord_q_clipped = torch.Tensor([float(x1_n - x1_q) / width_q, float(y1_n - y1_q) / height_q,
                                    float(x2_n - x1_q) / width_q, float(y2_n - y1_q) / height_q])
    coord_k_clipped = torch.Tensor([float(x1_n - x1_k) / width_k, float(y1_n - y1_k) / height_k,
                                    float(x2_n - x1_k) / width_k, float(y2_n - y1_k) / height_k])
    return [coord_q_clipped, coord_k_clipped]

@ALBPIPELINES.register_module(force=True)
class ToTensorV2(alb.BasicTransform):
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}

@ALBPIPELINES.register_module(force=True) 
class ColorJitter(alb.ImageOnlyTransform):
    r"""
    Randomly change brightness, contrast, hue and saturation of the image. This
    class behaves exactly like :class:`torchvision.transforms.ColorJitter` but
    is slightly faster (uses OpenCV) and compatible with rest of the transforms
    used here (albumentations-style). This class works only on ``uint8`` images.

    .. note::

        Unlike torchvision variant, this class follows "garbage-in, garbage-out"
        policy and does not check limits for jitter factors. User must ensure
        that ``brightness``, ``contrast``, ``saturation`` should be ``float``
        in ``[0, 1]`` and ``hue`` should be a ``float`` in ``[0, 0.5]``.

    Parameters
    ----------
    brightness: float, optional (default = 0.4)
        How much to jitter brightness. ``brightness_factor`` is chosen
        uniformly from ``[1 - brightness, 1 + brightness]``.
    contrast: float, optional (default = 0.4)
        How much to jitter contrast. ``contrast_factor`` is chosen uniformly
        from ``[1 - contrast, 1 + contrast]``
    saturation: float, optional (default = 0.4)
        How much to jitter saturation. ``saturation_factor`` is chosen
        uniformly from ``[1 - saturation, 1 + saturation]``.
    hue: float, optional (default = 0.4)
        How much to jitter hue. ``hue_factor`` is chosen uniformly from
        ``[-hue, hue]``.
    always_apply: bool, optional (default = False)
        Indicates whether this transformation should be always applied.
    p: float, optional (default = 0.5)
        Probability of applying the transform.
    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.4,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, img, **params):
        original_dtype = img.dtype

        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        # Convert arguments as required by albumentations functional interface.
        # "gain" = contrast and "bias" = (brightness_factor - 1)
        img = alb.augmentations.functional.brightness_contrast_adjust(
            img, alpha=contrast_factor, beta=brightness_factor - 1
        )
        # Hue and saturation limits are required to be integers.
        img = alb.augmentations.functional.shift_hsv(
            img,
            hue_shift=int(hue_factor * 255),
            sat_shift=int(saturation_factor * 255),
            val_shift=0,
        )
        img = img.astype(original_dtype)
        return img

    def get_transform_init_args_names(self):
        return ("brightness", "contrast", "saturation", "hue")

@ALBPIPELINES.register_module(force=True)
class CustomTwoCrop(object):
    def __init__(self, size=224, scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=TF.InterpolationMode.BILINEAR,
                condition_overlap=True):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation

        self.scale = scale
        self.ratio = ratio
        self.condition_overlap = condition_overlap

    @staticmethod
    def get_params(img, scale, ratio, ):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def get_params_conditioned(self, img, scale, ratio, constraint):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
            constraints list(tuple): list of params (i, j, h, w) that should be used to constrain the crop
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width
        for counter in range(10):
            rand_scale = random.uniform(*scale)
            target_area = rand_scale * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                intersection = _compute_intersection((i, j, h, w), constraint)
                if intersection >= 0.01 * target_area: # at least 1 percent of the second crop is part of the first crop.
                    return i, j, h, w, True
        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, False
        # return self.get_params(img, scale, ratio) # Fallback to default option

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            crops (list of lists): result of multi-crop
        """
        crops, coords = [], []
        params1 = self.get_params(img, self.scale, self.ratio)
        coords.append(_get_coord(*params1))
        crops.append(TF.resized_crop(img, *params1, self.size, self.interpolation))

        if not self.condition_overlap:
            params2 = self.get_params(img, self.scale, self.ratio)
        else:
            params2_ = self.get_params_conditioned(img, self.scale, self.ratio, params1)
            params2, passed = params2_[:4], params2_[-1]
            # if passed is true, the above func fell back to centercrop,
            # make two crops both centercrops in this case.
            
            if not passed:
                coords[0] = _get_coord(*params2)
                crops[0] = TF.resized_crop(img, *params2, self.size, self.interpolation)
        coords.append(_get_coord(*params2))
        crops.append(TF.resized_crop(img, *params2, self.size, self.interpolation))

        return crops, coords, _clip_coords(coords, [params1, params2])

@ALBPIPELINES.register_module(force=True)
class CustomRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, crops, coords, coords_clipped):
        crops_flipped, coords_flipped, coords_c_flipped, flags_flipped = [], [], [], []
        for crop, coord, coord_c in zip(crops, coords, coords_clipped):
            crop_flipped = crop
            coord_flipped = coord
            coord_c_flipped = coord_c

            flag_flipped = False
            if torch.rand(1) < self.p:
                crop_flipped = TF.hflip(crop)

                coord_flipped = coord.clone()
                coord_flipped[0] = coord[2]
                coord_flipped[2] = coord[0]

                coord_c_flipped = coord_c.clone()
                coord_c_flipped[0] = 1. - coord_c[2]
                coord_c_flipped[2] = 1. - coord_c[0]
                flag_flipped = True

            crops_flipped.append(crop_flipped)
            coords_flipped.append(coord_flipped)
            coords_c_flipped.append(coord_c_flipped)
            flags_flipped.append(flag_flipped)

        return crops_flipped, coords_flipped, coords_c_flipped, flags_flipped