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


class _GenBioTransform:
    def __init__(self):
        import torchvision.transforms as T
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.697, 0.575, 0.728), std=(0.188, 0.240, 0.187)),
        ])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_floating_point():
            x = (x.clamp(0, 1) * 255).byte()
        return self._transform(x)


class _HibouTransform:
    """Wraps HuggingFace AutoImageProcessor as a callable transform."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        img_np = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        inputs = self.processor(images=img_np, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)


class _HibouWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).last_hidden_state[:, 0]


class _Virchow2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        class_token  = out[:, 0]
        patch_tokens = out[:, 5:]
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


def load_feature_extractor(model_name: str, device: torch.device):
    """
    Loads a feature extractor by name and returns (model, transform, feat_dim).

    Supported model_name values:
        'uni2h'   -> MahmoodLab/UNI2-h  (timm, gated)
        'hibou-b'  -> histai/hibou-b      (transformers, public)
        'hibou-l'  -> histai/hibou-L      (transformers, public)
        'virchow2' -> paige-ai/Virchow2   (timm, gated)
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

    elif model_name == 'h-optimus-1':
        model = timm.create_model(
            'hf-hub:bioptimus/H-optimus-1', pretrained=True,
            init_values=1e-5, dynamic_img_size=False,
        ).to(device)
        model.eval()
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        feat_dim  = 1536

    elif model_name == 'virchow2':
        base_model = timm.create_model(
            'hf-hub:paige-ai/Virchow2', pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        ).to(device)
        base_model.eval()
        model     = _Virchow2Wrapper(base_model)
        transform = create_transform(**resolve_data_config(base_model.pretrained_cfg, model=base_model))
        feat_dim  = 2560

    elif model_name == 'genbio':
        from genbio_pathfm.model import GenBio_PathFM_Inference
        from config import GENBIO_WEIGHTS_PATH
        model     = GenBio_PathFM_Inference(GENBIO_WEIGHTS_PATH, device=str(device))
        transform = _GenBioTransform()
        feat_dim  = 4608

    else:
        raise ValueError(f'Unknown model: {model_name}. Choose from: {", ".join(["uni2h", "hibou-b", "hibou-l", "virchow2", "h-optimus-1", "genbio"])}')

    print(f'{model_name} loaded on {device}  (feat_dim={feat_dim})')
    return model, transform, feat_dim


def build_linear_probe(feat_dim: int, device: torch.device) -> torch.nn.Module:
    return torch.nn.Linear(feat_dim, 2).to(device)
