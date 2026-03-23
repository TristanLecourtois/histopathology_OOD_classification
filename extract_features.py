import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH, OUTPUT_DIR,
    EXTRACT_BATCH_SIZE, N_AUG, N_TTA, SEED,
)
from dataset import HistoDataset
from model import load_feature_extractor
from transforms import make_aug_transform

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    token = UserSecretsClient().get_secret('HF_TOKEN')
    login(token=token, add_to_git_credential=False)
    print('Logged in to HuggingFace.')
except Exception:
    print('HuggingFace secret not found, model must already be cached.')

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

feature_extractor, base_transform = load_feature_extractor(device)
aug_transform = make_aug_transform(base_transform)


@torch.no_grad()
def extract(h5_path: str, transform, mode: str, desc: str) -> dict:
    dataset = HistoDataset(h5_path, transform, mode)
    loader  = DataLoader(
        dataset,
        batch_size=EXTRACT_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    feats, labels, ids = [], [], []
    for imgs, ys, img_ids in tqdm(loader, desc=desc):
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            f = feature_extractor(imgs.to(device))
        feats.append(f.cpu())
        labels.append(ys if isinstance(ys, torch.Tensor) else torch.tensor(ys))
        ids.extend(img_ids.tolist())
    return {
        'features': torch.cat(feats,  dim=0),
        'labels':   torch.cat(labels, dim=0),
        'ids':      ids,
    }


print('Extracting val features...')
d = extract(VAL_PATH, base_transform, 'train', 'val')
torch.save(d, os.path.join(OUTPUT_DIR, 'val.pt'))
print(f'val: {d["features"].shape}')

print('Extracting train features (base)...')
d = extract(TRAIN_PATH, base_transform, 'train', 'train base')
torch.save(d, os.path.join(OUTPUT_DIR, 'train_base.pt'))
print(f'train_base: {d["features"].shape}')

for i in range(N_AUG):
    print(f'Extracting train features (aug {i+1}/{N_AUG})...')
    d = extract(TRAIN_PATH, aug_transform, 'train', f'train aug {i+1}')
    torch.save(d, os.path.join(OUTPUT_DIR, f'train_aug_{i}.pt'))
    print(f'train_aug_{i}: {d["features"].shape}')

print('Extracting test features (base)...')
d = extract(TEST_PATH, base_transform, 'test', 'test base')
torch.save(d, os.path.join(OUTPUT_DIR, 'test_base.pt'))
print(f'test_base: {d["features"].shape}')

for i in range(N_TTA):
    print(f'Extracting test features (tta {i+1}/{N_TTA})...')
    d = extract(TEST_PATH, aug_transform, 'test', f'test tta {i+1}')
    torch.save(d, os.path.join(OUTPUT_DIR, f'test_tta_{i}.pt'))
    print(f'test_tta_{i}: {d["features"].shape}')

print('Done. Files saved to', OUTPUT_DIR)
