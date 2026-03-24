import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from config import TIMM_KWARGS, FEAT_DIM


def load_feature_extractor(device: torch.device):
    model = timm.create_model(
        'hf-hub:MahmoodLab/UNI2-h',
        pretrained=True,
        **TIMM_KWARGS,
    ).to(device)
    model.eval()

    base_transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    print(f'UNI2-h loaded on {device}')
    return model, base_transform


def build_linear_probe(device: torch.device) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(FEAT_DIM, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 2),
    ).to(device)
