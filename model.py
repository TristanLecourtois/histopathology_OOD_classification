import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# UNI2-h architecture kwargs
_UNI2H_KWARGS = {
    'img_size': 224,
    'patch_size': 14,
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5,
    'embed_dim': 1536,
    'mlp_ratio': 2.66667 * 2,
    'num_classes': 0,
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked,
    'act_layer': torch.nn.SiLU,
    'reg_tokens': 8,
    'dynamic_img_size': True,
}


class _HibouTransform:
    """Wraps HuggingFace AutoImageProcessor as a callable transform."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        img_np = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        inputs = self.processor(images=img_np, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)


class _HibouWrapper(torch.nn.Module):
    """Wraps Hibou AutoModel to return CLS token features directly."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state[:, 0]


def load_feature_extractor(model_name: str, device: torch.device):
    """
    Loads a feature extractor by name and returns (model, transform, feat_dim).

    Supported model_name values:
        'uni2h'   -> MahmoodLab/UNI2-h  (timm, gated)
        'hibou-b' -> histai/hibou-b      (transformers, public)
        'hibou-l' -> histai/hibou-L      (transformers, public)
    """
    if model_name == 'uni2h':
        model = timm.create_model(
            'hf-hub:MahmoodLab/UNI2-h', pretrained=True, **_UNI2H_KWARGS
        ).to(device)
        model.eval()
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        feat_dim = 1536

    elif model_name in ('hibou-b', 'hibou-l'):
        from transformers import AutoImageProcessor, AutoModel
        hf_id = 'histai/hibou-b' if model_name == 'hibou-b' else 'histai/hibou-L'
        processor = AutoImageProcessor.from_pretrained(hf_id, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(device)
        base_model.eval()
        model     = _HibouWrapper(base_model)
        transform = _HibouTransform(processor)
        feat_dim  = 768 if model_name == 'hibou-b' else 1024

    else:
        raise ValueError(f'Unknown model: {model_name}. Choose from: uni2h, hibou-b, hibou-l')

    print(f'{model_name} loaded on {device}  (feat_dim={feat_dim})')
    return model, transform, feat_dim


def build_linear_probe(feat_dim: int, device: torch.device) -> torch.nn.Module:
    return torch.nn.Linear(feat_dim, 2).to(device)
