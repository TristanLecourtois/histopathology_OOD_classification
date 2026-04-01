import h5py
import numpy as np
import matplotlib.pyplot as plt
import stainlib.utils.stain_utils as stain_utils
from stainlib.normalization.normalizer import ReinhardStainNormalizer

TRAIN_PATH = '/kaggle/input/competitions/mva-dlmi-2026-histopathology-ood-classification/train.h5'
OUT_PATH   = '/kaggle/working/stain_normalization.pdf'

REF_ID     = '16'
CENTERS    = [0, 3, 4]
N_PER_CENTER = 2

def load_one_per_center(h5_path, centers, n_per_center):
    samples = {c: [] for c in centers}
    with h5py.File(h5_path, 'r') as hdf:
        for img_id in hdf.keys():
            center = int(np.array(hdf[img_id]['metadata'])[0])
            if center not in samples or len(samples[center]) >= n_per_center:
                continue
            img = np.array(hdf[img_id]['img'])
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            samples[center].append(img.transpose(1, 2, 0))
            if all(len(v) >= n_per_center for v in samples.values()):
                break
    return samples

with h5py.File(TRAIN_PATH, 'r') as hdf:
    ref = np.array(hdf[REF_ID]['img'])
    if ref.max() <= 1.0:
        ref = (ref * 255).clip(0, 255).astype(np.uint8)
    else:
        ref = ref.astype(np.uint8)
    ref_hwc = ref.transpose(1, 2, 0)

ref_std = stain_utils.LuminosityStandardizer.standardize(ref_hwc)
normalizer = ReinhardStainNormalizer()
normalizer.fit(ref_std)

def normalize(img_hwc):
    try:
        std = stain_utils.LuminosityStandardizer.standardize(img_hwc)
        return normalizer.transform(std, mask_background=False)
    except Exception:
        return img_hwc

samples = load_one_per_center(TRAIN_PATH, CENTERS, N_PER_CENTER)

n_cols = len(CENTERS) * N_PER_CENTER
fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

col = 0
for center in CENTERS:
    for img in samples[center]:
        axes[0, col].imshow(img)
        axes[0, col].set_title(f'Center {center} — original', fontsize=10)
        axes[0, col].axis('off')

        axes[1, col].imshow(normalize(img))
        axes[1, col].set_title(f'Center {center} — Reinhard', fontsize=10)
        axes[1, col].axis('off')
        col += 1

plt.suptitle('Histopathology patches before and after Reinhard stain normalization',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches='tight', dpi=150)
print(f'Saved: {OUT_PATH}')
plt.show()
