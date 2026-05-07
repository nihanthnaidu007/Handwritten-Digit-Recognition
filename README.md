# Handwritten Digit Recognition — Multi-Model ML Pipeline

**AI 681 — Machine Learning | Spring 2026 | Long Island University, Brooklyn**  
**Team:** Nihanth Naidu K, Likitha P, Shreya V

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-orange?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8%2B-blue)
![Datasets](https://img.shields.io/badge/Datasets-MNIST%20%7C%20EMNIST%20%7C%20USPS-lightgrey)
![License](https://img.shields.io/badge/License-Academic-purple)

---

## Table of Contents

- [Overview](#overview)
- [Final Results](#final-results)
- [Model Architectures](#model-architectures)
- [Pipeline Workflow](#pipeline-workflow)
- [Visual Results](#visual-results)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Setup Instructions](#setup-instructions)
- [Running the Pipeline](#running-the-pipeline)
- [Step Reference](#step-reference)
- [Configuration & Hyperparameters](#configuration--hyperparameters)
- [Cross-Dataset Transfer Analysis](#cross-dataset-transfer-analysis)
- [Weighted Soft Ensemble](#weighted-soft-ensemble)
- [GradCAM Explainability](#gradcam-explainability)
- [Live Demo](#live-demo)
- [Output Files](#output-files)
- [Reproducibility](#reproducibility)
- [Requirements](#requirements)
- [References](#references)
- [Course Information](#course-information)

---

## Overview

This project builds a **complete end-to-end machine learning pipeline** for handwritten digit recognition, benchmarking six models across three real-world datasets. It is structured as an 11-step automated pipeline — run one command and everything executes in order, from raw data download to an interactive web demo.

We go beyond single-dataset evaluation by testing **cross-dataset generalization**: models trained on MNIST are evaluated on EMNIST and USPS, revealing how well learned representations transfer to unseen distributions.

### What Makes This Different

- **6 models compared head-to-head:** Logistic Regression, KNN, Random Forest, SVM, CNN Baseline, CNN Improved — all trained and evaluated on the same splits
- **3 datasets:** MNIST (70K samples), EMNIST-Digits (280K samples), USPS (9.3K samples, 16×16 resized to 28×28)
- **Cross-dataset transfer matrix:** Every model trained on Dataset A is evaluated on Datasets B and C
- **Weighted soft ensemble:** Accuracy-weighted probability averaging across all six models
- **GradCAM explainability:** Gradient-weighted class activation maps showing which pixels drove each CNN prediction
- **Bias-variance analysis:** Sweep across hyperparameter values (k for KNN, trees for RFC, C for LR) to visualize underfitting vs. overfitting
- **Live Gradio demo:** Draw a digit in the browser — the CNN predicts it instantly with a GradCAM heatmap overlay

---

## Final Results

### Accuracy by Dataset

| Model | MNIST | EMNIST | USPS |
|-------|-------|--------|------|
| **CNN Improved** | **99.55%** | **99.69%** | **91.83%** |
| CNN Baseline | 99.28% | 99.67% | 90.98% |
| SVM | 97.87% | 96.66% | 87.29% |
| Random Forest | 97.05% | 97.29% | 89.44% |
| KNN | 96.88% | 96.11% | 86.40% |
| Logistic Regression | 92.56% | 94.22% | 86.95% |
| **Weighted Ensemble** | **99.55%+** | — | — |

### Key Observations

- The **CNN Improved model achieves 99.55% on MNIST** — matching near state-of-the-art for this architecture class
- **EMNIST accuracy (99.69%) is slightly higher than MNIST**, because EMNIST-Digits has 4× more training data (240K vs 60K)
- **USPS accuracy drops ~7–14 percentage points** across all models, revealing a meaningful domain gap (different scanner resolution, writing style, and contrast distribution)
- **MNIST → USPS cross-domain drop:** ~14.5 percentage points for the CNN Improved, confirming that domain shift is the dominant challenge
- **MNIST → EMNIST cross-domain drop:** Less than 0.2 percentage points — both datasets share the same distribution, so transfer is nearly lossless
- **Classical models (SVM, RFC)** are competitive but require no GPU and train in minutes

---

## Model Architectures

### CNN Baseline

A 3-layer convolutional network following the architecture introduced in lecture.

```
Input: (N, 1, 28, 28)
Block 1: Conv2d(1→32,  k=3, pad=1) → ReLU → MaxPool(2) → Dropout2d(0.20)  [28×28 → 14×14]
Block 2: Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool(2) → Dropout2d(0.25)  [14×14 → 7×7]
Block 3: Conv2d(64→128,k=3, pad=1) → ReLU → MaxPool(2) → Dropout2d(0.40)  [7×7 → 3×3]
Flatten: 128 × 3 × 3 = 1,152
Classifier: Linear(1152→128) → ReLU → Dropout(0.30) → Linear(128→10)
```

### CNN Improved

A deeper network with **BatchNormalization**, **dual conv blocks per stage**, **L2 regularization** (via AdamW weight decay), and tuned dropout — demonstrating the regularization techniques from Lecture 6.

```
Input: (N, 1, 28, 28)

Block 1: Conv2d(1→32,  k=3, pad=1) → BN → ReLU              [28×28]
         Conv2d(32→32, k=3, pad=1) → BN → ReLU → MaxPool(2)  [28×28 → 14×14]
         Dropout2d(0.25)

Block 2: Conv2d(32→64, k=3, pad=1) → BN → ReLU              [14×14]
         Conv2d(64→64, k=3, pad=1) → BN → ReLU → MaxPool(2)  [14×14 → 7×7]
         Dropout2d(0.25)

Flatten: 64 × 7 × 7 = 3,136

Classifier:
  Linear(3136→256) → ReLU → Dropout(0.40)
  Linear(256→128)  → ReLU → Dropout(0.30)
  Linear(128→10)
```

> **BatchNorm ordering:** Conv → BN → ReLU (BN before activation, before pooling — canonical order per He et al. 2016).

### Classical Models

| Model | Library | Key Hyperparameters |
|-------|---------|-------------------|
| Logistic Regression | scikit-learn | solver=lbfgs, C=1.0, max_iter=1000 |
| KNN | scikit-learn | k=5, algorithm=auto |
| Random Forest | scikit-learn | n_estimators=100 |
| SVM | scikit-learn | kernel=poly, gamma=0.1 |

> **Sample caps:** EMNIST is large (240K training samples). KNN is capped at 20K and SVM at 10K for EMNIST to keep runtimes reasonable. MNIST and USPS use all samples. These caps are configurable in `config.py`.

---

## Pipeline Workflow

The full pipeline from raw data to interactive demo:

```
Step 01  Data Download & Prep
         ↓ Downloads MNIST, EMNIST, USPS via torchvision
         ↓ Saves 3 formats per dataset: raw (28×28), flat (784,), cnn (1×28×28 float)
         ↓ USPS resized from 16×16 → 28×28 (PIL LANCZOS)

Step 02  Logistic Regression
         ↓ Trained on flat features, evaluated on all 3 datasets

Step 03  KNN
         ↓ Trained on flat features, evaluated on all 3 datasets

Step 04  Random Forest
         ↓ Trained on flat features, evaluated on all 3 datasets

Step 05  SVM (poly kernel)
         ↓ Trained on flat features, evaluated on all 3 datasets

Step 06  CNN Baseline
         ↓ 3-block CNN, trained on all 3 datasets independently

Step 07  CNN Improved
         ↓ BatchNorm + L2 + deeper blocks, trained on all 3 datasets
         ↓ Includes CNNNoDropout variant to demonstrate overfitting

Step 08  Bias-Variance Analysis
         ↓ Sweeps KNN (k=1→20), RFC (trees=10→200), LR (C=0.001→100)
         ↓ Plots train vs. val curves to show underfitting/overfitting regimes

Step 09  Model Comparison + Ensemble
         ↓ Side-by-side accuracy bar charts across all datasets
         ↓ Cross-dataset transfer matrix (train on A, test on B)
         ↓ Weighted soft ensemble + cost-sensitive classification

Step 10  Visualizations
         ↓ GradCAM heat maps (one per digit class, per dataset)
         ↓ CNN filter visualizations, feature maps, t-SNE embeddings
         ↓ Misclassification grids

Step 11  Live Demo
         ↓ Gradio browser app: draw digit → CNN prediction + GradCAM overlay
         ↓ Dataset selector: choose MNIST / EMNIST / USPS model
```

---

## Visual Results

All plots are generated automatically to `outputs/plots/` at 150 DPI during the pipeline run.

### Model Comparison
Per-dataset accuracy bar charts comparing all 6 models side by side (MNIST / EMNIST / USPS panels).

### Cross-Dataset Transfer Matrix
A heatmap showing accuracy when each model trained on Dataset A is evaluated on Dataset B — makes domain gap immediately visible.

### Bias-Variance Curves
Train vs. validation accuracy plotted across:
- KNN: k = 1, 3, 5, 7, 10, 15, 20
- Random Forest: trees = 10, 25, 50, 100, 200
- Logistic Regression: C = 0.001, 0.01, 0.1, 1.0, 10.0, 100.0

### GradCAM Heatmaps
Per-digit class GradCAM overlays on test samples, generated for all three datasets. Shows which pixels in the 28×28 image activated the CNN's prediction most strongly (hooks on `features[10]`, the second conv of Block 2).

### t-SNE Embeddings
2D projection of CNN feature vectors (extracted before the final classifier layer), colored by digit class. Shows natural clustering in feature space.

### Misclassification Grids
Grid of test samples the CNN got wrong, with predicted vs. true labels shown — reveals systematic failure modes.

---

## Repository Structure

```
mnist_project_v2/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── .gitignore                         ← Excludes outputs/, venv/, __pycache__/
│
├── run.py                             ← Master pipeline runner (start here)
├── config.py                          ← All paths, dataset names, hyperparameters
├── models.py                          ← CNN class definitions (CNNBaseline, CNNImproved, CNNNoDropout)
├── utils.py                           ← Shared helpers: logging, loaders, GradCAM, t-SNE, plots
│
├── pipeline/
│   ├── __init__.py
│   ├── step_01_data.py                ← Download MNIST, EMNIST, USPS; save 3 formats each
│   ├── step_02_logistic_regression.py ← LR training + evaluation across datasets
│   ├── step_03_knn.py                 ← KNN training + evaluation
│   ├── step_04_rfc.py                 ← Random Forest training + evaluation
│   ├── step_05_svm.py                 ← SVM training + evaluation
│   ├── step_06_cnn_baseline.py        ← 3-block CNN, per-dataset training
│   ├── step_07_cnn_improved.py        ← Deeper CNN + BN + dropout + L2, per-dataset training
│   ├── step_08_bias_variance.py       ← Hyperparameter sweep, bias-variance curves
│   ├── step_09_comparison.py          ← All-model comparison, ensemble, transfer matrix
│   ├── step_10_visualisations.py      ← GradCAM, filters, t-SNE, misclassification grids
│   └── step_11_demo.py                ← Gradio live drawing demo
│
└── outputs/                           ← Auto-created; everything generated goes here
    ├── data/
    │   ├── mnist.npz                  ← Raw (N, 28, 28) uint8
    │   ├── mnist_flat.npz             ← Flattened (N, 784) for sklearn
    │   ├── mnist_cnn.npz              ← (N, 1, 28, 28) float32 for PyTorch
    │   ├── emnist.npz / emnist_flat.npz / emnist_cnn.npz
    │   └── usps.npz  / usps_flat.npz  / usps_cnn.npz
    ├── models/
    │   ├── logistic_regression_{mnist,emnist,usps}.pkl
    │   ├── knn_{mnist,emnist,usps}.pkl
    │   ├── rfc_{mnist,emnist,usps}.pkl
    │   ├── svm_{mnist,emnist,usps}.pkl
    │   ├── cnn_baseline_{mnist,emnist,usps}.pt
    │   └── cnn_improved_{mnist,emnist,usps}.pt
    ├── plots/
    │   ├── model_comparison.png       ← Side-by-side accuracy across datasets
    │   ├── transfer_matrix.png        ← Cross-dataset transfer heatmap
    │   ├── bias_variance_knn.png
    │   ├── bias_variance_rfc.png
    │   ├── bias_variance_lr.png
    │   ├── gradcam_{mnist,emnist,usps}.png
    │   ├── tsne_{mnist,emnist,usps}.png
    │   ├── filters_cnn_improved.png
    │   └── misclassified_{mnist,emnist,usps}.png
    ├── reports/
    │   ├── logistic_regression_{mnist,emnist,usps}.json
    │   ├── knn_{mnist,emnist,usps}.json
    │   ├── rfc_{mnist,emnist,usps}.json
    │   ├── svm_{mnist,emnist,usps}.json
    │   ├── cnn_baseline_{mnist,emnist,usps}.json
    │   ├── cnn_improved_{mnist,emnist,usps}.json
    │   ├── bias_variance.json
    │   └── comparison.json
    └── logs/
        └── step_01.log ... step_11.log
```

> **Note:** The `outputs/` folder is auto-created on first run and is excluded from git via `.gitignore`. The `venv/` folder is also excluded.

---

## Datasets

All datasets are downloaded automatically by Step 01. No manual download is needed.

| Dataset | Train Samples | Test Samples | Image Size | Source |
|---------|--------------|-------------|------------|--------|
| MNIST | 60,000 | 10,000 | 28×28 grayscale | torchvision |
| EMNIST-Digits | 240,000 | 40,000 | 28×28 grayscale | torchvision |
| USPS | 7,291 | 2,007 | 16×16 → resized to 28×28 | torchvision |

### Preprocessing Applied

- **EMNIST:** Images are stored transposed in torchvision's HDF5 file — a standard `np.transpose` fix is applied to restore the correct orientation
- **USPS:** Originally 16×16 pixels. Resized to 28×28 using PIL LANCZOS resampling so all datasets share the same input resolution. SSL verification is temporarily disabled for the USPS download (the USPS server has a certificate chain that Windows/Anaconda cannot verify)
- **Normalization for CNNs:** Raw pixel values (0–255 uint8) are divided by 255.0 to produce float [0, 1]. Classical sklearn models use the raw flattened uint8 values

### Dataset Format on Disk

Each dataset is saved in three `.npz` formats:

| File suffix | Shape | dtype | Used by |
|------------|-------|-------|---------|
| `.npz` (raw) | `(N, 28, 28)` | uint8 | Visualizations |
| `_flat.npz` | `(N, 784)` | uint8 | sklearn classical models |
| `_cnn.npz` | `(N, 1, 28, 28)` | float32 | PyTorch CNN training |

---

## Setup Instructions

### Prerequisites

- **Python 3.9+** (tested on 3.10 and 3.11)
- **pip** package manager
- **~3 GB free disk space** (datasets + model checkpoints + plots)
- **(Recommended) NVIDIA GPU** with CUDA 12.8 — CPU works but CNN training will be much slower

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd mnist_project_v2
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux / macOS:
source venv/bin/activate
```

### Step 3: Install PyTorch (GPU Users — Do This First)

If you have an NVIDIA GPU, install the CUDA-compatible PyTorch build **before** installing the rest of the requirements. Check your CUDA version first:

```bash
nvidia-smi   # look for "CUDA Version: XX.X" in the top-right corner
```

Then install the matching build:

```bash
# CUDA 12.8 (RTX 40/50 series — Blackwell, Ada Lovelace):
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.1 (RTX 30 series and older):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (no GPU):
pip install torch torchvision
```

### Step 4: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify GPU Detection

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Running on CPU')"
```

Expected output with a GPU present:

```
CUDA: True
NVIDIA GeForce RTX 5060
```

You are now ready to run the pipeline.

---

## Running the Pipeline

### Run Everything (Recommended)

```bash
python run.py
```

This runs all 11 steps in sequence. Steps that detect their outputs already exist will skip automatically, so you can safely re-run after an interruption without starting over.

### Run From a Specific Step

All steps from the given number onward will run in sequence:

```bash
python run.py --from 6     # Run Steps 6 through 11
python run.py --from 9     # Just comparison, visualizations, and demo
```

### Run a Single Step in Isolation

Useful for debugging or regenerating specific outputs without re-running the whole pipeline:

```bash
python run.py --only 7     # CNN Improved only
python run.py --only 10    # Regenerate all plots
python run.py --only 11    # Launch the demo
```

### List All Steps

```bash
python run.py --list
```

---

## Step Reference

| Step | Script | What It Does | Approx. Time |
|------|--------|-------------|-------------|
| 1 | `step_01_data.py` | Download MNIST, EMNIST, USPS; save raw / flat / cnn formats | < 2 min |
| 2 | `step_02_logistic_regression.py` | Train LR on all 3 datasets, save models + JSON reports | 2–3 min |
| 3 | `step_03_knn.py` | Train KNN on all 3 datasets (EMNIST capped at 20K) | 10–15 min |
| 4 | `step_04_rfc.py` | Train Random Forest on all 3 datasets | 2–3 min |
| 5 | `step_05_svm.py` | Train SVM on all 3 datasets (EMNIST capped at 10K) | 15–20 min |
| 6 | `step_06_cnn_baseline.py` | Train 3-block CNN on all 3 datasets | 3–5 min (GPU) |
| 7 | `step_07_cnn_improved.py` | Train improved CNN (BN + L2 + dropout) on all 3 datasets | 5–8 min (GPU) |
| 8 | `step_08_bias_variance.py` | Sweep KNN / RFC / LR hyperparameters, plot bias-variance curves | 10–15 min |
| 9 | `step_09_comparison.py` | All-model accuracy comparison, ensemble, transfer matrix | 3–5 min |
| 10 | `step_10_visualisations.py` | GradCAM, filter plots, t-SNE, misclassification grids | 5–8 min |
| 11 | `step_11_demo.py` | Launch Gradio web demo (stays running until Ctrl+C) | Instant |

> **GPU vs CPU:** Times shown assume an NVIDIA RTX GPU. CPU-only runtimes will be 5–10× longer for the CNN steps, and longer still for KNN and SVM on large datasets.

---

## Configuration & Hyperparameters

All settings live in `config.py`. Change a value there and re-run from the affected step:

```bash
# Example: changed KNN k from 5 to 7 in config.py
python run.py --from 3
```

### CNN Hyperparameters

| Hyperparameter | CNN Baseline | CNN Improved | Rationale |
|----------------|-------------|-------------|-----------|
| Optimizer | Adam | AdamW | AdamW adds decoupled weight decay (L2 regularization) |
| Learning Rate | 1e-3 | 1e-3 | Standard starting point for CNN training |
| Batch Size | 256 | 256 | Maximizes GPU throughput on 28×28 images |
| Epochs | 20 | 30 | Improved model needs more epochs to converge |
| Early Stopping Patience | — | 5 epochs | Stops if validation loss doesn't improve |
| Validation Split | — | 20% | 20% of training data held out for early stopping |
| Dropout (conv layers) | 0.20–0.40 | 0.25 | Spatial dropout applied to feature maps |
| Dropout (dense layers) | 0.30 | 0.40 | Standard dense-layer dropout |
| Weight Decay | — | 1e-3 | L2 penalty via AdamW optimizer |
| BatchNorm | ✗ | ✓ | Stabilizes training, reduces LR sensitivity |

### Classical Model Hyperparameters

| Model | Parameter | Value |
|-------|-----------|-------|
| Logistic Regression | solver | lbfgs |
| | max_iter | 1000 |
| | C (inverse regularization strength) | 1.0 |
| | random_state | 42 |
| KNN | n_neighbors (k) | 5 |
| | algorithm | auto |
| | n_jobs | −1 (use all CPU cores) |
| Random Forest | n_estimators | 100 |
| | n_jobs | −1 (use all CPU cores) |
| | random_state | 42 |
| SVM | kernel | poly |
| | gamma | 0.1 |

### Sample Caps for Large Datasets

| Model | MNIST | EMNIST | USPS |
|-------|-------|--------|------|
| LR | All | All | All |
| KNN | All | 20,000 | All |
| RFC | All | 60,000 | All |
| SVM | All | 10,000 | All |

### Bias-Variance Sweep Values

| Model | Parameter Swept | Values |
|-------|----------------|--------|
| KNN | k (neighbors) | 1, 3, 5, 7, 10, 15, 20 |
| Random Forest | n_estimators | 10, 25, 50, 100, 200 |
| Logistic Regression | C | 0.001, 0.01, 0.1, 1.0, 10.0, 100.0 |

Sweeps are run on a 10,000-sample subset for speed. Each point evaluates both train and validation accuracy to show underfitting and overfitting regimes clearly.

---

## Cross-Dataset Transfer Analysis

Step 09 builds a **transfer matrix** — each model is trained on one dataset and evaluated on all three. Key results for CNN Improved:

| Trained on → Tested on | Accuracy | Drop from In-Domain |
|------------------------|----------|-------------------|
| MNIST → MNIST | 99.55% | — (in-domain) |
| MNIST → EMNIST | ~99.3% | < 0.2pp |
| MNIST → USPS | ~85.1% | −14.5pp |
| EMNIST → EMNIST | 99.69% | — (in-domain) |
| USPS → USPS | 91.83% | — (in-domain) |

**Why USPS is hard:** USPS images were originally scanned at 16×16 resolution with different contrast, stroke thickness, and writing style compared to MNIST. Even after upscaling to 28×28, the distribution shift causes a ~14.5 percentage point accuracy drop.

**Why EMNIST transfers almost perfectly:** EMNIST-Digits was collected using the same pipeline as MNIST (NIST Special Database 19) and shares the same resolution, centering, and normalization. The domain gap is essentially zero.

---

## Weighted Soft Ensemble

Step 09 builds an accuracy-weighted ensemble combining all six models:

1. Each model produces a **probability vector** over 10 classes for every test sample
2. For classical sklearn models, `predict_proba()` is used directly
3. For SVM (which lacks calibrated probabilities by default), `decision_function()` is converted to probabilities via softmax
4. Each model's probability output is **weighted by its validation-set accuracy**
5. The weighted average probability vector is computed and the highest-probability class is the prediction

This ensemble consistently matches or exceeds the best individual model (CNN Improved) on MNIST.

---

## GradCAM Explainability

GradCAM (Gradient-weighted Class Activation Mapping) is applied to the CNN Improved model in Step 10 to show which image regions drove each prediction.

**How it works:**

1. A forward pass is run on a test image
2. Gradients of the predicted class score are computed with respect to the feature maps at `features[10]` — the second Conv2d of Block 2 (64→64 channels)
3. Gradients are global-average-pooled across the spatial dimension to produce per-channel importance weights
4. A weighted sum of the feature maps is computed, and ReLU is applied (keeping only positive activations)
5. The resulting importance map is upsampled to 28×28 and overlaid on the original image as a color heatmap

GradCAM is run on one example per digit class (0–9) for each of the three datasets, producing three output plots:

```
outputs/plots/gradcam_mnist.png
outputs/plots/gradcam_emnist.png
outputs/plots/gradcam_usps.png
```

---

## Live Demo

Step 11 launches an interactive browser-based application using Gradio:

```bash
python run.py --only 11
# Open http://127.0.0.1:7860 in your browser
```

**Features:**

- **Drawing canvas:** Draw any digit (0–9) with your mouse or touchscreen
- **Dataset selector:** Choose which trained model to use — MNIST, EMNIST, or USPS
- **Prediction output:** Displays the predicted digit and confidence score instantly
- **GradCAM overlay:** Shows a heatmap of which parts of your drawing drove the prediction
- **USPS preprocessing:** When the USPS model is selected, the drawn image is internally downsampled to 16×16 then upscaled to 28×28 to match the distribution the USPS model was trained on

> **Prerequisite:** Step 7 must have completed successfully before launching the demo. All three CNN Improved model files must exist in `outputs/models/`.

---

## Output Files

After the full pipeline completes, the following files exist in `outputs/`:

```
outputs/
├── data/
│   ├── mnist.npz, mnist_flat.npz, mnist_cnn.npz
│   ├── emnist.npz, emnist_flat.npz, emnist_cnn.npz
│   └── usps.npz,  usps_flat.npz,  usps_cnn.npz
│
├── models/
│   ├── logistic_regression_{mnist,emnist,usps}.pkl
│   ├── knn_{mnist,emnist,usps}.pkl
│   ├── rfc_{mnist,emnist,usps}.pkl
│   ├── svm_{mnist,emnist,usps}.pkl
│   ├── cnn_baseline_{mnist,emnist,usps}.pt
│   └── cnn_improved_{mnist,emnist,usps}.pt       ← 6 files total (one per dataset)
│
├── plots/
│   ├── model_comparison.png                       ← All 6 models × 3 datasets
│   ├── transfer_matrix.png                        ← Cross-dataset transfer heatmap
│   ├── bias_variance_knn.png                      ← k vs. train/val accuracy
│   ├── bias_variance_rfc.png                      ← n_estimators vs. train/val accuracy
│   ├── bias_variance_lr.png                       ← C vs. train/val accuracy
│   ├── gradcam_{mnist,emnist,usps}.png            ← GradCAM per digit class, per dataset
│   ├── tsne_{mnist,emnist,usps}.png               ← t-SNE feature space projection
│   ├── filters_cnn_improved.png                   ← Learned conv filter visualizations
│   └── misclassified_{mnist,emnist,usps}.png      ← CNN error grid per dataset
│
├── reports/
│   ├── logistic_regression_{mnist,emnist,usps}.json  ← Accuracy, per-class metrics
│   ├── knn_{mnist,emnist,usps}.json
│   ├── rfc_{mnist,emnist,usps}.json
│   ├── svm_{mnist,emnist,usps}.json
│   ├── cnn_baseline_{mnist,emnist,usps}.json
│   ├── cnn_improved_{mnist,emnist,usps}.json
│   ├── bias_variance.json                            ← Full sweep results
│   └── comparison.json                               ← Ensemble and transfer results
│
└── logs/
    └── step_01.log ... step_11.log                   ← Full execution log per step
```

---

## Reproducibility

All random seeds are fixed:

| Component | Seed |
|-----------|------|
| NumPy | 42 |
| PyTorch (CPU + CUDA) | 13 |
| scikit-learn models | 42 |

The CNN training uses `random_state=13` for weight initialization reproducibility.

> **Note on GPU reproducibility:** CUDA operations are not fully deterministic by default due to non-deterministic cuDNN algorithms. Results may vary by ±0.05% accuracy between GPU runs, but are consistent across CPU runs.

---

## Requirements

```
numpy==2.4.4
matplotlib==3.10.9
scikit-learn==1.8.0
scipy==1.17.1
pillow==12.2.0
joblib==1.5.3
gradio==6.14.0
```

**PyTorch must be installed separately** (see [Setup Instructions](#setup-instructions)):

```
# GPU (CUDA 12.8):
torch==2.11.0+cu128
torchvision==0.26.0+cu128

# CPU:
torch (latest)
torchvision (latest)
```

**Python version:** 3.9 or higher (tested on 3.10 and 3.11)

---

## Quick Start (TL;DR)

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd mnist_project_v2

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install PyTorch with GPU support (adjust version as needed)
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Run the full pipeline
python run.py

# 6. Launch the interactive drawing demo when the pipeline finishes
python run.py --only 11
# → Open http://127.0.0.1:7860 in your browser and draw a digit
```

**Total runtime:** ~60–90 minutes on GPU (KNN and SVM dominate due to dataset size), ~4–6 hours on CPU.

---

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278–2324.

2. Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). *EMNIST: Extending MNIST to handwritten letters.* International Joint Conference on Neural Networks (IJCNN).

3. Hull, J.J. (1994). *A database for handwritten text recognition research.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 16(5), 550–554. (USPS Dataset)

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Identity Mappings in Deep Residual Networks.* ECCV 2016. (BatchNorm → ReLU canonical ordering)

5. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.

6. Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML 2015.

7. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.

8. Cortes, C. & Vapnik, V. (1995). *Support-Vector Networks.* Machine Learning, 20(3), 273–297.

---

## Course Information

| | |
|--|--|
| **Course** | AI 681 — Machine Learning |
| **Semester** | Spring 2026 |
| **Institution** | Long Island University, Brooklyn |
| **Team** | Nihanth Naidu K, Likitha P, Shreya V |

---

## License

This project is for academic purposes as part of the AI 681 coursework at Long Island University. The MNIST, EMNIST, and USPS datasets are publicly available through their respective sources and torchvision.
