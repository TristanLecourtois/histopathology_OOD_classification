import argparse
import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from tqdm import tqdm

from config import (
    OUTPUT_DIR, TRAIN_PATH, LR, MOMENTUM, WEIGHT_DECAY,
    NUM_EPOCHS, PATIENCE, TRAIN_BS, SEED, SUPPORTED_MODELS,
    DANN_LAMBDA, DANN_GAMMA,
)
from dann import DANNModel, get_alpha
from mixstyle import MixStyle

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='uni2h', choices=SUPPORTED_MODELS)
parser.add_argument('--mixstyle', action='store_true')
args = parser.parse_args()

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = os.path.join(OUTPUT_DIR, args.model)
print(f'Device: {device}  Model: {args.model}')


def load(name):
    return torch.load(os.path.join(model_dir, name), weights_only=True)


# Load precomputed train features
train_dicts = [load('train_base.pt')]
i = 0
while os.path.exists(os.path.join(model_dir, f'train_aug_{i}.pt')):
    train_dicts.append(load(f'train_aug_{i}.pt'))
    i += 1

# Read center (domain) labels from h5 using saved image IDs
base_ids = train_dicts[0]['ids']
with h5py.File(TRAIN_PATH, 'r') as hdf:
    center_list = [int(np.array(hdf[str(img_id)]['metadata'])[0]) for img_id in base_ids]

unique_centers = sorted(set(center_list))
center_map     = {c: i for i, c in enumerate(unique_centers)}
num_domains    = len(unique_centers)
print(f'Centers found: {unique_centers}  ->  {num_domains} domains')

base_centers = torch.tensor([center_map[c] for c in center_list], dtype=torch.long)

# Build datasets with (features, labels, domain_labels)
train_datasets = []
for d in train_dicts:
    features = d['features']
    labels   = d['labels'].long()
    train_datasets.append(TensorDataset(features, labels, base_centers))

train_ds = ConcatDataset(train_datasets)
val_dict = load('val.pt')
val_ds   = TensorDataset(val_dict['features'], val_dict['labels'].long())

feat_dim = train_dicts[0]['features'].shape[1]
print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  feat_dim: {feat_dim}')

train_loader = DataLoader(train_ds, batch_size=TRAIN_BS, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=256,      shuffle=False, num_workers=2)

mixstyle    = MixStyle(p=0.5, alpha=0.1).to(device) if args.mixstyle else None
model       = DANNModel(feat_dim, num_domains).to(device)
optimizer   = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
label_crit  = nn.CrossEntropyLoss()
domain_crit = nn.CrossEntropyLoss()
acc_metric  = torchmetrics.Accuracy('multiclass', num_classes=2)

best_val_acc = 0.0
best_epoch   = 0
save_path    = os.path.join(model_dir, 'best_dann.pth')

for epoch in range(NUM_EPOCHS):
    model.train()
    alpha = get_alpha(epoch, NUM_EPOCHS, gamma=DANN_GAMMA)
    train_losses, t_preds, t_trues = [], [], []

    for x, y, d in tqdm(train_loader, desc=f'Epoch {epoch+1:3d} train', leave=False):
        x, y, d = x.to(device), y.to(device), d.to(device)
        optimizer.zero_grad()
        if mixstyle is not None:
            x = mixstyle(x)
        label_logits, domain_logits = model(x, alpha=alpha)
        loss = label_crit(label_logits, y) + DANN_LAMBDA * domain_crit(domain_logits, d)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        t_preds.append(label_logits.detach().cpu())
        t_trues.append(y.cpu())

    train_acc = acc_metric(torch.cat(t_preds), torch.cat(t_trues)).item()

    model.eval()
    val_losses, v_preds, v_trues = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            label_logits, _ = model(x, alpha=0.0)
            val_losses.append(label_crit(label_logits, y).item())
            v_preds.append(label_logits.cpu())
            v_trues.append(y.cpu())

    val_acc  = acc_metric(torch.cat(v_preds), torch.cat(v_trues)).item()
    val_loss = float(np.mean(val_losses))

    marker = ''
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        torch.save(model.state_dict(), save_path)
        marker = '  <- best'

    print(
        f'Epoch {epoch+1:3d}/{NUM_EPOCHS}  alpha={alpha:.3f}'
        f'  train_loss={np.mean(train_losses):.4f}  train_acc={train_acc:.4f}'
        f'  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}'
        + marker
    )

    if epoch - best_epoch >= PATIENCE:
        print(f'Early stopping (best epoch={best_epoch+1})')
        break

print(f'Model saved: {save_path}')
print(f'Best epoch: {best_epoch+1}  val_acc: {best_val_acc:.4f}')
