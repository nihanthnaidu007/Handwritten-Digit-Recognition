"""Step 01 - Download and prepare MNIST, EMNIST, and USPS datasets.

Three datasets saved in three formats each:
  raw  (N, 28, 28) uint8           - for visualisation
  flat (N, 784)    uint8           - for sklearn classical models
  cnn  (N, 1, 28, 28) float32/255  - PyTorch NCHW for CNN training

EMNIST (split='digits'): 240k/40k samples, 28x28.
  Images are stored rotated in torchvision -- standard transpose fix applied.
USPS: 7291/2007 samples, originally 16x16.
  Resized to 28x28 (PIL LANCZOS) so all datasets share the same resolution.
  USPS is hosted on a server whose cert chain Windows/Anaconda cannot verify.
  SSL verification is temporarily disabled only for that download call.
"""

import ssl
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config
from utils import get_logger


def save_dataset(name, X_train, y_train, X_test, y_test, log):
    np.savez_compressed(
        config.data_file(name),
        X_train=X_train,
        y_train=y_train.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
    )
    log.info("  [%s] raw   saved  shape=%s", name, str(X_train.shape))

    np.savez_compressed(
        config.flat_file(name),
        X_train=X_train.reshape(-1, 784),
        y_train=y_train.astype(np.int64),
        X_test=X_test.reshape(-1, 784),
        y_test=y_test.astype(np.int64),
    )
    log.info("  [%s] flat  saved  shape=%s", name, str(X_train.reshape(-1, 784).shape))

    X_tr_cnn = X_train[:, np.newaxis, :, :].astype(np.float32) / 255.0
    X_te_cnn = X_test[:,  np.newaxis, :, :].astype(np.float32) / 255.0
    np.savez_compressed(
        config.cnn_file(name),
        X_train=X_tr_cnn,
        y_train=y_train.astype(np.int64),
        X_test=X_te_cnn,
        y_test=y_test.astype(np.int64),
    )
    log.info("  [%s] cnn   saved  shape=%s", name, str(X_tr_cnn.shape))


def load_mnist(log):
    from torchvision.datasets import MNIST
    log.info("Downloading MNIST via torchvision...")
    train_ds = MNIST(root=str(config.DATA_DIR), train=True,  download=True)
    test_ds  = MNIST(root=str(config.DATA_DIR), train=False, download=True)
    X_train  = train_ds.data.numpy()
    y_train  = train_ds.targets.numpy()
    X_test   = test_ds.data.numpy()
    y_test   = test_ds.targets.numpy()
    log.info("  MNIST  train=%s  test=%s", str(X_train.shape), str(X_test.shape))
    return X_train, y_train, X_test, y_test


def load_emnist(log):
    """EMNIST digits split. Applies transpose fix for torchvision orientation."""
    from torchvision.datasets import EMNIST
    log.info("Downloading EMNIST (digits split) via torchvision...")
    train_ds = EMNIST(root=str(config.DATA_DIR), split="digits",
                      train=True,  download=True)
    test_ds  = EMNIST(root=str(config.DATA_DIR), split="digits",
                      train=False, download=True)
    X_train = train_ds.data.numpy()
    y_train = train_ds.targets.numpy()
    X_test  = test_ds.data.numpy()
    y_test  = test_ds.targets.numpy()
    # Fix orientation: torchvision EMNIST is stored rotated relative to MNIST
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test  = np.transpose(X_test,  (0, 2, 1))
    log.info("  EMNIST train=%s  test=%s", str(X_train.shape), str(X_test.shape))
    return X_train, y_train, X_test, y_test


def load_usps(log):
    """USPS: 16x16 float32 in [-1, 1]. Convert to uint8 and resize to 28x28.

    USPS is hosted at csie.ntu.edu.tw whose TLS certificate chain is not
    trusted by the default Windows/Anaconda CA bundle. We temporarily replace
    the default HTTPS context with an unverified one for the download only,
    then restore the original context immediately after.
    """
    from torchvision.datasets import USPS
    from PIL import Image

    log.info("Downloading USPS via torchvision...")

    # Check if already downloaded to avoid unnecessary SSL bypass
    usps_train_path = Path(config.DATA_DIR) / "usps" / "usps.bz2"
    usps_test_path  = Path(config.DATA_DIR) / "usps" / "usps.t.bz2"
    already_cached  = usps_train_path.exists() and usps_test_path.exists()

    if not already_cached:
        log.info("  USPS not cached -- bypassing SSL for download (Windows CA bundle issue).")
        _orig_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            train_ds = USPS(root=str(config.DATA_DIR), train=True,  download=True)
            test_ds  = USPS(root=str(config.DATA_DIR), train=False, download=True)
        finally:
            # Always restore -- even if download fails
            ssl._create_default_https_context = _orig_ctx
        log.info("  SSL context restored.")
    else:
        log.info("  USPS already cached -- loading without download.")
        train_ds = USPS(root=str(config.DATA_DIR), train=True,  download=False)
        test_ds  = USPS(root=str(config.DATA_DIR), train=False, download=False)

    def process(dataset):
        raw = np.array(dataset.data)
        # Convert from [-1, 1] float to [0, 255] uint8
        raw = ((raw + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        # Resize 16x16 -> 28x28 with high-quality LANCZOS resampling
        resized = np.stack([
            np.array(Image.fromarray(img).resize((28, 28), Image.LANCZOS))
            for img in raw
        ])
        return resized, np.array(dataset.targets)

    X_train, y_train = process(train_ds)
    X_test,  y_test  = process(test_ds)
    log.info("  USPS   train=%s  test=%s", str(X_train.shape), str(X_test.shape))
    return X_train, y_train, X_test, y_test


def main():
    log = get_logger("step_01_data")
    log.info("Step 01 - Multi-Dataset Download & Preprocessing")
    log.info("Datasets: MNIST / EMNIST (digits) / USPS")

    X_tr, y_tr, X_te, y_te = load_mnist(log)
    save_dataset("mnist", X_tr, y_tr, X_te, y_te, log)

    X_tr, y_tr, X_te, y_te = load_emnist(log)
    save_dataset("emnist", X_tr, y_tr, X_te, y_te, log)

    X_tr, y_tr, X_te, y_te = load_usps(log)
    save_dataset("usps", X_tr, y_tr, X_te, y_te, log)

    log.info("Step 01 complete - all 9 .npz files saved.")


if __name__ == "__main__":
    main()
