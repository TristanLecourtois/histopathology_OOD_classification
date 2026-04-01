import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import torch
import torch.nn.functional as F

from config import OUTPUT_DIR, SUPPORTED_MODELS
from model import build_linear_probe

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='uni2h', choices=SUPPORTED_MODELS)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  Model: {args.model}')

model_dir = os.path.join(OUTPUT_DIR, args.model)


def load(name):
    return torch.load(os.path.join(model_dir, name), weights_only=True)


feat_dim = load('train_base.pt')['features'].shape[1]
linear_probe = build_linear_probe(feat_dim, device)
linear_probe.load_state_dict(load('best_linear_probe.pth'))
linear_probe.eval()

test_files = ['test_base.pt']
i = 0
while os.path.exists(os.path.join(model_dir, f'test_tta_{i}.pt')):
    test_files.append(f'test_tta_{i}.pt')
    i += 1

print(f'TTA views: {len(test_files)}')
test_ids = load('test_base.pt')['ids']

all_probs = []
with torch.no_grad():
    for fname in test_files:
        feats  = load(fname)['features'].to(device)
        logits = linear_probe(feats)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu()
        all_probs.append(probs)

mean_probs  = torch.stack(all_probs, dim=0).mean(dim=0)
final_preds = (mean_probs > 0.5).int().numpy()

submission = (
    pd.DataFrame({'ID': test_ids, 'Pred': final_preds})
    .set_index('ID')
    .sort_index()
)

out_path = os.path.join(OUTPUT_DIR, f'submission_{args.model}.csv')
submission.to_csv(out_path)
print(f'Saved: {out_path}  ({len(submission)} predictions)')
print(submission['Pred'].value_counts().to_string())
