# Privacy-Preserving Adaptive Behavioral Fusion Framework for Web Continuous Authentication

> **Paper:** "A Privacy-Preserving Adaptive Behavioral Fusion Framework for Web Continuous Authentication: Multi-User Evaluation on the BB-MAS Dataset"
> **Authors:** [Anonymized for Peer Review]
> **Institution:** [Anonymized Institution]

---

## Overview

This repository contains the full implementation of a client-side multimodal continuous authentication (CA) framework using enriched behavioral biometrics (keystroke, mouse, scroll) evaluated on the BB-MAS dataset.

**Key Results (80 users, LOUO, SMOTE):**

| Method | Mean EER | AUC |
|--------|----------|-----|
| Keystroke only (proposed) | 21.06% | 0.847 |
| Mouse only | 34.24% | 0.692 |
| Scroll only | 5.15%† | 0.499† |
| MLP Fusion K+M | 24.44% | 0.810 |

† Degenerate — class collapse artifact, not real performance.

---

## Repository Structure

```
├── src/
│   ├── bbmas_auth_final.py       # Main evaluation script (80-user LOUO)
│   └── bbmas_extensions.py       # Extension experiments (drift + adversarial)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes dataset and output files
└── README.md                     # This file
```

---

## Dataset

This project uses the **BB-MAS (Browser-Based Multi-modal Authentication Suite)** dataset.

- **Source:** IEEE Dataport — https://ieee-dataport.org/open-access/su-ais-bb-mas-syracuse-university-and-assured-information-security-behavioral
- **Citation:** M. Frank, T. Leitner, and M. Schurmann, "SU-AIS BB-MAS: A Multi-Modal Behavioral Authentication Dataset," IEEE Dataport, 2019.
- The dataset is **not included** in this repository. It is freely available as open access on IEEE Dataport — no subscription required, only a free IEEE account.

### Dataset Setup

1. Download the BB-MAS dataset from IEEE Dataport (requires free account)
2. Extract the zip file
3. Set the path in both scripts:

```python
# In bbmas_auth_final.py and bbmas_extensions.py
BBMAS_ROOT = r"path/to/BB-MAS_Dataset"
```

The expected folder structure is:
```
BB-MAS_Dataset/
├── 1/
│   ├── 1_Desktop_Keyboard.csv
│   ├── 1_Mouse_Move.csv
│   └── 1_Mouse_Wheel.csv
├── 2/
│   ├── 2_Desktop_Keyboard.csv
│   ...
└── 117/
    └── ...
```

---

## Installation

### Requirements
- Python 3.8+
- Windows / Linux / macOS

### Setup

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/bbmas-auth-framework-xxxx/
cd bbmas-auth-framework

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Run main evaluation (80 users, LOUO)

```bash
cd src
python bbmas_auth_final.py
```

This will:
- Load and chunk all three modalities from BB-MAS
- Train per-user MLP models with SMOTE
- Evaluate using Leave-One-User-Out (LOUO) protocol
- Output results table and LaTeX-ready tables
- Save `paper_per_user_results_final.csv`

Expected runtime: **3–5 hours** on CPU (80 users × 3 modalities × MLP training)

### Step 2 — Run extension experiments (drift + adversarial)

```bash
python bbmas_extensions.py
```

This will:
- Run **Experiment A**: Drift simulation with EMA adaptation (116 users)
- Run **Experiment B**: Adversarial mimicry robustness (116 users)
- Save `paper_drift_results.csv` and `paper_adversarial_results.csv`

Expected runtime: **2–4 hours** on CPU

---

## Reproducibility

The scripts use fixed seeds for reproducibility:

```python
os.environ['TF_DETERMINISTIC_OPS']   = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED']         = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

All impostor user lists are sorted before selection to ensure consistent held-out user assignment across runs.

---

## Feature Dimensions

| Modality | Dimensions | Key Features |
|----------|-----------|--------------|
| Keystroke | 20 | Dwell mean/std/CV/skew/kurt/entropy, Bigram mean/std, FFT₁₂₃, Flight mean/std/CV/skew/kurt/entropy, Typing rate, Key count |
| Mouse | 14 | Speed mean/std/max/skew/kurt/entropy, FFT₁₂₃, Accel mean/std, Path length, Dir changes, Path efficiency |
| Scroll | 10 | Delta mean/std/skew/kurt/entropy, ITI mean/std, Frequency, Burst ratio, Event count |

---

## Model Architecture

```
Input (d-dim)
    → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    → Dense(64, ReLU)  → BatchNorm → Dropout(0.3)
    → Dense(32, ReLU)
    → Dense(1, Sigmoid)
```

Trained with:
- Adam optimizer (lr=1e-3)
- Binary cross-entropy loss
- SMOTE oversampling on training set
- Balanced class weights
- Early stopping (patience=8, monitor=val_loss)

---

## Expected Output

After running `bbmas_auth_final.py`, the console prints:

```
=================================================================
  FINAL RESULTS  (80 users | CHUNK_MOUSE=50 | LOUO | SMOTE)
=================================================================

  Method                                     EER      Std     AUC   vs 37.54%
  ------------------------------------------------------------------------------
  Keystroke only (20-dim, enriched)        21.06%   10.42%  0.847    +43.9%
  Mouse only (14-dim, chunk=50)            34.24%   15.39%  0.692     +8.8%
  ...
```

---

## Citation

[Citation removed for double-blind review]

---

## License

This code is released for academic research purposes. See `LICENSE` for details.
