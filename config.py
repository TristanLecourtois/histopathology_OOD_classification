import os

DATA_DIR   = '/kaggle/input/competitions/mva-dlmi-2026-histopathology-ood-classification'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.h5')
VAL_PATH   = os.path.join(DATA_DIR, 'val.h5')
TEST_PATH  = os.path.join(DATA_DIR, 'test.h5')
OUTPUT_DIR = '/kaggle/working'

SUPPORTED_MODELS = ['uni2h', 'hibou-b', 'hibou-l', 'virchow2', 'h-optimus-1']

EXTRACT_BATCH_SIZE = 64
N_AUG = 3
N_TTA = 5

LR           = 2e-4
MOMENTUM     = 0.9
WEIGHT_DECAY = 0
NUM_EPOCHS   = 50
PATIENCE     = 10
TRAIN_BS     = 8

SEED = 0

DANN_LAMBDA = 1.0
DANN_GAMMA  = 10.0
