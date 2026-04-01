import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TRAIN_PATH = '/kaggle/input/competitions/mva-dlmi-2026-histopathology-ood-classification/train.h5'
VAL_PATH   = '/kaggle/input/competitions/mva-dlmi-2026-histopathology-ood-classification/val.h5'
OUT_PATH   = '/kaggle/working/rgb_distribution.pdf'

N_SAMPLES_PER_CENTER = 300

center_pixels = {}

for path, expected_centers in [(TRAIN_PATH, [0, 3, 4]), (VAL_PATH, [1])]:
    with h5py.File(path, 'r') as hdf:
        ids = list(hdf.keys())
        np.random.shuffle(ids)
        collected = {c: [] for c in expected_centers}
        for img_id in ids:
            center = int(np.array(hdf[img_id]['metadata'])[0])
            if center not in collected:
                continue
            if len(collected[center]) >= N_SAMPLES_PER_CENTER:
                continue
            img = np.array(hdf[img_id]['img'])  # (3, H, W), float or uint
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            # flatten spatial dims, keep channels: (3, H*W)
            collected[center].append(img.reshape(3, -1))
            if all(len(v) >= N_SAMPLES_PER_CENTER for v in collected.values()):
                break
        for c, arrays in collected.items():
            if arrays:
                center_pixels[c] = np.concatenate(arrays, axis=1)  # (3, N)

colors_rgb  = ['#e74c3c', '#2ecc71', '#3498db']
channel_names = ['Red', 'Green', 'Blue']
center_colors = {0: '#1a1a2e', 3: '#e94560', 4: '#0f3460', 1: '#533483'}
center_labels = {0: 'Center 0 (train)', 3: 'Center 3 (train)', 4: 'Center 4 (train)', 1: 'Center 1 (val)'}

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
bins = np.linspace(0, 255, 64)

for ch_idx, (ax, ch_name, ch_color) in enumerate(zip(axes, channel_names, colors_rgb)):
    for center, pixels in sorted(center_pixels.items()):
        vals = pixels[ch_idx]
        ax.hist(vals, bins=bins, density=True, alpha=0.55,
                color=center_colors[center], label=center_labels[center],
                histtype='stepfilled', linewidth=1.2, edgecolor=center_colors[center])
    ax.set_title(f'{ch_name} channel', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pixel intensity', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_xlim(0, 255)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [mpatches.Patch(color=center_colors[c], label=center_labels[c])
           for c in sorted(center_pixels.keys())]
axes[-1].legend(handles=handles, fontsize=10, loc='upper left')

plt.suptitle('RGB intensity distributions across centers', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches='tight', dpi=150)
print(f'Saved: {OUT_PATH}')
plt.show()
