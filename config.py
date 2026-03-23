import os
import torch
import timm

DATA_DIR   = '/kaggle/input/mva-dlmi-2026-histopathology-ood-classification'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.h5')
VAL_PATH   = os.path.join(DATA_DIR, 'val.h5')
TEST_PATH  = os.path.join(DATA_DIR, 'test.h5')
OUTPUT_DIR = '/kaggle/working'

TIMM_KWARGS = {
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

FEAT_DIM = 1536

EXTRACT_BATCH_SIZE = 16
N_AUG = 3
N_TTA = 5

LR           = 2e-4
MOMENTUM     = 0.9
WEIGHT_DECAY = 0
NUM_EPOCHS   = 50
PATIENCE     = 10
TRAIN_BS     = 8

SEED = 0
