"""Step 04 - Random Forest Classifier on MNIST, EMNIST, and USPS."""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config
from utils import get_logger, load_flat_ds, plot_confusion_matrix, save_report


def train_on_dataset(ds, log):
    log.info(f"\n--- Dataset: {ds.upper()} ---")
    X_train, y_train, X_test, y_test = load_flat_ds(ds)

    cap = config.CLASSICAL_SAMPLE_CAP["rfc"][ds]
    if cap and len(X_train) > cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), cap, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        log.info(f"  Training capped at {cap:,} samples")

    log.info(f"  Training RFC {config.RFC_PARAMS}  n_train={len(X_train):,}")
    clf = RandomForestClassifier(**config.RFC_PARAMS)
    clf.fit(X_train, y_train)

    joblib.dump(clf, config.rfc_model(ds))
    log.info(f"  Model saved -> {config.rfc_model(ds).name}")

    y_pred   = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True)
    log.info(f"  Test accuracy [{ds}]: {test_acc:.4f}")

    plot_confusion_matrix(
        cm,
        title=f"Random Forest [{ds.upper()}]  acc={test_acc*100:.2f}%",
        save_path=config.PLOTS_DIR / f"confusion_matrix_rfc_{ds}.png",
    )

    importance = clf.feature_importances_.reshape(28, 28)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(importance, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Feature importance")
    ax.set_title(f"RFC Pixel Importance [{ds.upper()}]\n(which pixels matter most?)", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / f"feature_importance_rfc_{ds}.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    save_report(
        {"model": "Random Forest", "dataset": ds, "test_accuracy": test_acc,
         "confusion_matrix": cm, "classification_report": report,
         "params": config.RFC_PARAMS, "feature_importances": clf.feature_importances_},
        config.rfc_report(ds),
    )


def main():
    log = get_logger("step_04_rfc")
    for ds in config.DATASETS:
        train_on_dataset(ds, log)
    log.info("\nStep 04 complete.")


if __name__ == "__main__":
    main()
