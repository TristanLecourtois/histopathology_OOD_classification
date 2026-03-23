import random

import numpy as np
import torch
import torchvision.transforms as T
from stainlib.augmentation.augmenter import (
    HedLightColorAugmenter,
    HedLighterColorAugmenter,
    HedStrongColorAugmenter,
)


def _hed_augment(x: torch.Tensor) -> torch.Tensor:
    img_np = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    aug = random.choice([HedLightColorAugmenter, HedLighterColorAugmenter, HedStrongColorAugmenter])()
    aug.randomize()
    img_aug = aug.transform(img_np)
    return torch.tensor(img_aug, dtype=torch.float32).permute(2, 0, 1) / 255.0


def make_aug_transform(base_transform):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=90)], p=0.5),
        T.RandomApply([T.Lambda(_hed_augment)], p=0.5),
        base_transform,
    ])
