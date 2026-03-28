# Histopathology OOD Classification

## Setup (Kaggle)

```python
!git clone https://github.com/TristanLecourtois/-histopathology_OOD_classification.git /kaggle/working/project
!pip install -r /kaggle/working/project/requirements.txt -q
!pip install git+https://github.com/sebastianffx/stainlib.git -q
```

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
```

---

## Pipeline

**Step 1 — Feature extraction**
Loads the foundation model and runs it on train, val and test images to extract embeddings.
Applies optional Reinhard stain normalization and HED augmentation.
Results are saved as `.pt` files in `/kaggle/working/{model}/`.
If interrupted, relaunch the same command — already computed files are skipped.
```python
!HF_TOKEN=$HF_TOKEN python /kaggle/working/project/extract_features.py --model MODEL [options]
```

**Step 2 — Training**
Trains a linear classifier on top of the precomputed embeddings.
Saves the best model weights based on validation accuracy.
```python
!python /kaggle/working/project/train.py --model MODEL
```

**Step 3 — Prediction**
Loads the trained classifier, runs it on all test embeddings (with TTA averaging),
and saves the final predictions to `/kaggle/working/submission_{MODEL}.csv`.
```python
!python /kaggle/working/project/predict.py --model MODEL
```

---

## Models

| Name          | Gated | feat_dim |
|---------------|-------|----------|
| `uni2h`       | yes   | 1536     |
| `hibou-b`     | no    | 768      |
| `hibou-l`     | yes   | 1024     |
| `virchow2`    | yes   | 2560     |
| `h-optimus-1` | yes   | 1536     |

---

## Options — extract_features.py

| Argument        | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `--model`       | `uni2h` | Model backbone                       |
| `--n-aug`       | `3`     | Augmentation passes on train         |
| `--n-tta`       | `5`     | TTA passes on test                   |
| `--no-reinhard` | off     | Disable Reinhard stain normalization |

---

## DANN — Domain Adversarial Training

**What it is:**
DANN forces the feature extractor to produce embeddings that are indistinguishable across hospitals (centers).
It does this by adding a domain classifier that tries to predict which center an image comes from,
and a gradient reversal layer (GRL) that flips the gradients from this classifier.
The label classifier is thus trained on features that are both discriminative for tumor/non-tumor AND blind to the center.

**How it works:**
- The label classifier is trained normally (minimize label loss)
- The domain classifier tries to predict the center (0, 3 or 4 in train)
- The GRL reverses its gradients → the shared features are pushed to be domain-invariant
- Alpha (λ) starts at 0 and increases during training so domain adaptation kicks in gradually

**Uses the same extracted features** — no need to re-run `extract_features.py`.

**Commands:**
```python
!python /kaggle/working/project/dann_train.py --model MODEL
!python /kaggle/working/project/dann_predict.py --model MODEL
```

Submission saved to `/kaggle/working/submission_{MODEL}_dann.csv`.
