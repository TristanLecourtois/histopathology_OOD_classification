import random

import numpy as np
import torch
import torchvision.transforms as T
import stainlib.utils.stain_utils as stain_utils
from stainlib.augmentation.augmenter import (
    HedLightColorAugmenter,
    HedLighterColorAugmenter,
    HedStrongColorAugmenter,
)
from stainlib.normalization.normalizer import ReinhardStainNormalizer


def _hed_augment(x: torch.Tensor) -> torch.Tensor:
    img_np = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    aug = HedStrongColorAugmenter()
    aug.randomize()
    img_aug = aug.transform(img_np)
    return torch.tensor(img_aug, dtype=torch.float32).permute(2, 0, 1) / 255.0


class ReinhardTransform:
    """
    Normalizes staining of an image towards a reference image using Reinhard method.
    Reduces inter-center color shift before feature extraction.

    ref_img: CHW float [0,1] numpy array or tensor (image ID '16' from train set)
    """

    def __init__(self, ref_img: np.ndarray):
        target = (ref_img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        target = stain_utils.LuminosityStandardizer.standardize(target)
        self._normalizer = ReinhardStainNormalizer()
        self._normalizer.fit(target)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        img_np = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img_np = stain_utils.LuminosityStandardizer.standardize(img_np)
        img_norm = self._normalizer.transform(img_np, mask_background=False)
        return torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1) / 255.0


def make_base_transform(timm_transform, reinhard: ReinhardTransform = None):
    steps = []
    if reinhard is not None:
        steps.append(T.Lambda(reinhard))
    steps.append(timm_transform)
    return T.Compose(steps)


def make_aug_transform(timm_transform, reinhard: ReinhardTransform = None):
    steps = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=90)], p=0.5),
        T.RandomApply([T.Lambda(_hed_augment)], p=0.8),
    ]
    if reinhard is not None:
        steps.append(T.Lambda(reinhard))
    steps.append(timm_transform)
    return T.Compose(steps)
