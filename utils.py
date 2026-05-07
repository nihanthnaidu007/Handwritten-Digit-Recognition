"""Shared utilities used by every step."""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import config


# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    log_path = config.LOGS_DIR / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    try:
        utf8_stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        ch = logging.StreamHandler(utf8_stdout)
    except Exception:
        ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ── JSON report helpers ───────────────────────────────────────────────────────
def save_report(data: dict, path: Path):
    def _convert(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_convert)
    print(f"  Report -> {path.relative_to(config.ROOT)}")


def load_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: Path):
    plt.style.use(config.PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.matshow(cm, cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title(title, pad=16, fontsize=13)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.tight_layout()
    fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot  -> {save_path.relative_to(config.ROOT)}")


def plot_training_curves(history: dict, title: str, save_path: Path):
    plt.style.use(config.PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["accuracy"],     label="Train",      marker="o", markersize=3)
    axes[0].plot(history["val_accuracy"], label="Validation", marker="o", markersize=3)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["loss"],     label="Train",      marker="o", markersize=3)
    axes[1].plot(history["val_loss"], label="Validation", marker="o", markersize=3)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot  -> {save_path.relative_to(config.ROOT)}")


# ── Data loaders (legacy single-dataset) ─────────────────────────────────────
def load_flat():
    return load_flat_ds("mnist")

def load_cnn():
    return load_cnn_ds("mnist")

def load_raw():
    return load_raw_ds("mnist")


# ── Data loaders (per-dataset) ────────────────────────────────────────────────
def load_flat_ds(ds: str):
    d = np.load(config.flat_file(ds))
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

def load_cnn_ds(ds: str):
    d = np.load(config.cnn_file(ds))
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

def load_raw_ds(ds: str):
    d = np.load(config.data_file(ds))
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


# ── PyTorch training helpers ──────────────────────────────────────────────────
def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(X_train, y_train, X_test, y_test, batch_size: int, val_split: float = 0.2):
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)
    full_ds  = TensorDataset(X_tr, y_tr)
    val_size = int(len(full_ds) * val_split)
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(full_ds, [trn_size, val_size],
                                  generator=torch.Generator().manual_seed(13))
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    tst_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    return trn_loader, val_loader, tst_loader


def make_augmented_loaders(X_train, y_train, X_test, y_test,
                           batch_size: int, val_split: float = 0.2):
    """
    Like make_loaders but applies data augmentation to training batches.
    Augmentation: random affine (rotation ±10°, translation ±10%, scale 90-110%)
    + random erasing (simulates pen-lift / occlusion).
    Validation and test sets are NOT augmented (clean evaluation).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
    from torchvision import transforms

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)

    full_ds  = TensorDataset(X_tr, y_tr)
    val_size = int(len(full_ds) * val_split)
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(full_ds, [trn_size, val_size],
                                  generator=torch.Generator().manual_seed(13))

    augment = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10),
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15),
                                 ratio=(0.3, 3.3), value=0),
    ])

    class AugDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset    = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            x = self.transform(x)
            return x, y

    trn_loader = DataLoader(AugDataset(trn_ds, augment),
                            batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    tst_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    return trn_loader, val_loader, tst_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (out.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    import torch
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct    += (out.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)
    return total_loss / total, correct / total


def predict_all(model, loader, device):
    import torch
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = model(X_batch.to(device))
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(y_batch.numpy())
    return np.concatenate(preds), np.concatenate(trues)


# ── CNN inference helpers ──────────────────────────────────────────────────────
def load_cnn_improved(device, ds: str = "mnist"):
    """Load CNNImproved state_dict for a given dataset."""
    import torch
    from models import CNNImproved
    model = CNNImproved().to(device)
    model.load_state_dict(
        torch.load(config.cnn_imp(ds), map_location=device, weights_only=True)
    )
    model.eval()
    return model


def cnn_predict(model, X_flat: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    import torch
    X = X_flat.astype(np.float32)
    if X.max() > 1.0:
        X = X / 255.0
    X_cnn = torch.tensor(X.reshape(-1, 1, 28, 28))
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_cnn), batch_size):
            preds.append(model(X_cnn[i:i + batch_size].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def cnn_predict_proba(model, X_flat: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    X = X_flat.astype(np.float32)
    if X.max() > 1.0:
        X = X / 255.0
    X_cnn = torch.tensor(X.reshape(-1, 1, 28, 28))
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_cnn), batch_size):
            out = model(X_cnn[i:i + batch_size].to(device))
            probs.append(F.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(probs)


def cnn_extract_features(model, X_flat: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    """Extract 128-D penultimate features via forward hook on classifier[4]."""
    import torch
    X = X_flat.astype(np.float32)
    if X.max() > 1.0:
        X = X / 255.0
    X_cnn = torch.tensor(X.reshape(-1, 1, 28, 28))
    feats = []

    def _hook(module, inp, out):
        feats.append(out.detach().cpu().numpy())

    handle = model.classifier[4].register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_cnn), batch_size):
            model(X_cnn[i:i + batch_size].to(device))
    handle.remove()
    return np.concatenate(feats, axis=0)


# ── GradCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNNImproved.
    Highlights which spatial regions of the input drove the model's prediction.

    Usage:
        cam_engine = GradCAM(model, target_layer=model.features[-3])  # last conv
        heatmap, pred_class = cam_engine.compute(x_tensor)
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.activations  = None
        self.gradients    = None
        self._fwd_handle  = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle  = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def compute(self, x, class_idx=None):
        """
        x          : (1, 1, 28, 28) float32 tensor on the correct device.
        class_idx  : class to explain (None = argmax / predicted class).
        Returns    : (heatmap np.ndarray shape 28x28 in [0,1], predicted_class int)
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        x = x.clone().requires_grad_(True)

        out = self.model(x)
        pred_class = int(out.argmax(1).item()) if class_idx is None else class_idx

        self.model.zero_grad()
        out[0, pred_class].backward()

        # Global-average-pool the gradients → importance weight per channel
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam     = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, pred_class

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def make_tsne(n_components=2, random_state=42, perplexity=30, max_iter=1000):
    """Create TSNE — handles both old (n_iter) and new (max_iter) sklearn."""
    from sklearn.manifold import TSNE
    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 5):
        return TSNE(n_components=n_components, random_state=random_state,
                    perplexity=perplexity, max_iter=max_iter)
    else:
        return TSNE(n_components=n_components, random_state=random_state,
                    perplexity=perplexity, n_iter=max_iter)
