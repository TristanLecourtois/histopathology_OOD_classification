import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HistoDataset(Dataset):
    def __init__(self, h5_path: str, transform, mode: str = 'train'):
        self.h5_path   = h5_path
        self.transform = transform
        self.mode      = mode
        with h5py.File(h5_path, 'r') as hdf:
            self.image_ids = list(hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.h5_path, 'r') as hdf:
            img   = torch.tensor(np.array(hdf[img_id]['img']))
            label = int(np.array(hdf[img_id]['label'])) if self.mode != 'test' else -1
        return self.transform(img).float(), label, int(img_id)


class PrecomputedDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels   = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
