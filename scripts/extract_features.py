import argparse
import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH, OUTPUT_DIR,
    EXTRACT_BATCH_SIZE, N_AUG, N_TTA, SEED, SUPPORTED_MODELS,
)
from dataset import HistoDataset
from model import load_feature_extractor
from transforms import ReinhardTransform, make_base_transform, make_aug_transform

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='uni2h', choices=SUPPORTED_MODELS)
parser.add_argument('--no-reinhard', action='store_true')
parser.add_argument('--n-aug', type=int, default=N_AUG)
parser.add_argument('--n-tta', type=int, default=N_TTA)
args = parser.parse_args()

N_AUG = args.n_aug
N_TTA = args.n_tta

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
print(f'Device: {device}  Model: {args.model}')

model_dir = os.path.join(OUTPUT_DIR, args.model)
os.makedirs(model_dir, exist_ok=True)

feature_extractor, timm_transform, feat_dim = load_feature_extractor(args.model, device)

reinhard = None
if not args.no_reinhard:
    with h5py.File(TRAIN_PATH, 'r') as hdf:
        ref_img = np.array(hdf['16']['img'])
    reinhard = ReinhardTransform(ref_img)
    print('Reinhard normalizer fitted.')
else:
    print('Reinhard normalization disabled.')

base_transform = make_base_transform(timm_transform, reinhard)
aug_transform  = make_aug_transform(timm_transform, reinhard)


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


def save(name, data):
    torch.save(data, os.path.join(model_dir, name))


def exists(name):
    path = os.path.join(model_dir, name)
    if os.path.exists(path):
        print(f'Skipping {name} (already exists)')
        return True
    return False


if not exists('val.pt'):
    print('Extracting val...')
    d = extract(VAL_PATH, base_transform, 'train', 'val')
    save('val.pt', d)
    print(f'val: {d["features"].shape}')

if not exists('train_base.pt'):
    print('Extracting train (base)...')
    d = extract(TRAIN_PATH, base_transform, 'train', 'train base')
    save('train_base.pt', d)
    print(f'train_base: {d["features"].shape}')

for i in range(N_AUG):
    if not exists(f'train_aug_{i}.pt'):
        print(f'Extracting train (aug {i+1}/{N_AUG})...')
        d = extract(TRAIN_PATH, aug_transform, 'train', f'train aug {i+1}')
        save(f'train_aug_{i}.pt', d)
        print(f'train_aug_{i}: {d["features"].shape}')

if not exists('test_base.pt'):
    print('Extracting test (base)...')
    d = extract(TEST_PATH, base_transform, 'test', 'test base')
    save('test_base.pt', d)
    print(f'test_base: {d["features"].shape}')

for i in range(N_TTA):
    if not exists(f'test_tta_{i}.pt'):
        print(f'Extracting test (tta {i+1}/{N_TTA})...')
        d = extract(TEST_PATH, aug_transform, 'test', f'test tta {i+1}')
        save(f'test_tta_{i}.pt', d)
        print(f'test_tta_{i}: {d["features"].shape}')

print(f'Done. Files saved to {model_dir}')
