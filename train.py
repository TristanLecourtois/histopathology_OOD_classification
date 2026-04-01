import argparse
import os
import random

import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from config import (
    OUTPUT_DIR, LR, MOMENTUM, WEIGHT_DECAY,
    NUM_EPOCHS, PATIENCE, TRAIN_BS, SEED, SUPPORTED_MODELS,
)
from dataset import PrecomputedDataset
from model import build_linear_probe
from mixstyle import MixStyle

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='uni2h', choices=SUPPORTED_MODELS)
parser.add_argument('--mixstyle', action='store_true')
args = parser.parse_args()

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  Model: {args.model}')

model_dir = os.path.join(OUTPUT_DIR, args.model)


def load(name):
    return torch.load(os.path.join(model_dir, name), weights_only=True)


train_dicts = [load('train_base.pt')]
i = 0
while os.path.exists(os.path.join(model_dir, f'train_aug_{i}.pt')):
    train_dicts.append(load(f'train_aug_{i}.pt'))
    i += 1

train_ds = ConcatDataset([
    PrecomputedDataset(d['features'], d['labels']) for d in train_dicts
])

val_dict = load('val.pt')
val_ds   = PrecomputedDataset(val_dict['features'], val_dict['labels'])

feat_dim = train_dicts[0]['features'].shape[1]
print(f'Train: {len(train_ds)} samples ({len(train_dicts)} passes)  Val: {len(val_ds)}  feat_dim: {feat_dim}')

train_loader = DataLoader(train_ds, batch_size=TRAIN_BS, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=256,      shuffle=False, num_workers=2)

mixstyle     = MixStyle(p=0.5, alpha=0.1).to(device) if args.mixstyle else None
linear_probe = build_linear_probe(feat_dim, device)
optimizer    = torch.optim.SGD(linear_probe.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
criterion    = torch.nn.CrossEntropyLoss()
acc_metric   = torchmetrics.Accuracy('multiclass', num_classes=2)

best_val_acc = 0.0
best_epoch   = 0
save_path    = os.path.join(model_dir, 'best_linear_probe.pth')

for epoch in range(NUM_EPOCHS):
    linear_probe.train()
    train_losses, t_preds, t_trues = [], [], []
    for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1:3d} train', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if mixstyle is not None:
            x = mixstyle(x)
        logits = linear_probe(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        t_preds.append(logits.detach().cpu())
        t_trues.append(y.cpu())

    train_acc = acc_metric(torch.cat(t_preds), torch.cat(t_trues)).item()

    linear_probe.eval()
    val_losses, v_preds, v_trues = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = linear_probe(x)
            val_losses.append(criterion(logits, y).item())
            v_preds.append(logits.cpu())
            v_trues.append(y.cpu())

    val_acc  = acc_metric(torch.cat(v_preds), torch.cat(v_trues)).item()
    val_loss = float(np.mean(val_losses))

    marker = ''
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        torch.save(linear_probe.state_dict(), save_path)
        marker = '  <- best'

    print(
        f'Epoch {epoch+1:3d}/{NUM_EPOCHS}'
        f'  train_loss={np.mean(train_losses):.4f}  train_acc={train_acc:.4f}'
        f'  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}'
        + marker
    )

    if epoch - best_epoch >= PATIENCE:
        print(f'Early stopping (best epoch={best_epoch+1})')
        break

print(f'Model saved: {save_path}')
print(f'Best epoch: {best_epoch+1}  val_acc: {best_val_acc:.4f}')
