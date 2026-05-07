"""Step 03 - K-Nearest Neighbors on MNIST, EMNIST, and USPS."""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config
from utils import get_logger, load_flat_ds, load_raw_ds, plot_confusion_matrix, save_report


def train_on_dataset(ds, log):
    log.info(f"\n--- Dataset: {ds.upper()} ---")
    X_train, y_train, X_test, y_test = load_flat_ds(ds)

    cap = config.CLASSICAL_SAMPLE_CAP["knn"][ds]
    if cap and len(X_train) > cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), cap, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        log.info(f"  Training capped at {cap:,} samples (memory/speed)")

    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32)  / 255.0

    log.info(f"  Training KNN {config.KNN_PARAMS}  n_train={len(X_train):,}")
    clf = KNeighborsClassifier(**config.KNN_PARAMS)
    clf.fit(X_train, y_train)

    joblib.dump(clf, config.knn_model(ds))
    log.info(f"  Model saved -> {config.knn_model(ds).name}")

    log.info("  Predicting on test set...")
    y_pred   = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True)
    log.info(f"  Test accuracy [{ds}]: {test_acc:.4f}")

    plot_confusion_matrix(
        cm,
        title=f"KNN (k={config.KNN_PARAMS['n_neighbors']}) [{ds.upper()}]  acc={test_acc*100:.2f}%",
        save_path=config.PLOTS_DIR / f"confusion_matrix_knn_{ds}.png",
    )

    X_raw = load_raw_ds(ds)[2]
    plt.style.use(config.PLOT_STYLE)
    fig, axes = plt.subplots(2, 10, figsize=(18, 4))
    rng  = np.random.default_rng(42)
    idxs = rng.integers(0, len(y_test), 20)
    for ax, i in zip(axes.flat, idxs):
        ax.imshow(X_raw[i], cmap="gray", interpolation="nearest")
        color = "green" if y_pred[i] == y_test[i] else "red"
        ax.set_title(f"T:{y_test[i]} P:{y_pred[i]}", fontsize=8, color=color)
        ax.axis("off")
    fig.suptitle(f"KNN Sample Predictions [{ds.upper()}] (green=correct, red=wrong)", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / f"predictions_knn_{ds}.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    save_report(
        {"model": "KNN", "dataset": ds, "test_accuracy": test_acc,
         "confusion_matrix": cm, "classification_report": report,
         "params": config.KNN_PARAMS, "n_train": len(X_train)},
        config.knn_report(ds),
    )


def main():
    log = get_logger("step_03_knn")
    for ds in config.DATASETS:
        train_on_dataset(ds, log)
    log.info("\nStep 03 complete.")


if __name__ == "__main__":
    main()
