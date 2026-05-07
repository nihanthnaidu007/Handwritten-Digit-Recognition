"""Step 02 - Multinomial Logistic Regression on MNIST, EMNIST, and USPS."""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config
from utils import get_logger, load_flat_ds, plot_confusion_matrix, save_report


def train_on_dataset(ds, log):
    log.info(f"\n--- Dataset: {ds.upper()} ---")
    X_train, y_train, X_test, y_test = load_flat_ds(ds)

    cap = config.CLASSICAL_SAMPLE_CAP["lr"][ds]
    if cap and len(X_train) > cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), cap, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        log.info(f"  Training capped at {cap:,} samples")

    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32)  / 255.0

    params = {**config.LR_PARAMS, "max_iter": 2000}
    log.info(f"  Training LogisticRegression {params}")
    clf = LogisticRegression(**params)
    clf.fit(X_train, y_train)

    joblib.dump(clf, config.lr_model(ds))
    log.info(f"  Model saved -> {config.lr_model(ds).name}")

    y_pred   = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True)
    log.info(f"  Test accuracy [{ds}]: {test_acc:.4f}")

    plot_confusion_matrix(
        cm,
        title=f"Logistic Regression [{ds.upper()}]  acc={test_acc*100:.2f}%",
        save_path=config.PLOTS_DIR / f"confusion_matrix_logistic_regression_{ds}.png",
    )
    save_report(
        {"model": "Logistic Regression", "dataset": ds, "test_accuracy": test_acc,
         "confusion_matrix": cm, "classification_report": report, "params": params},
        config.lr_report(ds),
    )


def main():
    log = get_logger("step_02_logistic_regression")
    for ds in config.DATASETS:
        train_on_dataset(ds, log)
    log.info("\nStep 02 complete.")


if __name__ == "__main__":
    main()
