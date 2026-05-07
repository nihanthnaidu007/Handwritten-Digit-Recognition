"""Step 08 - Bias-Variance Analysis.

Classical sweeps use MNIST only (speed). CNN epoch curves loaded for all datasets,
showing how regularisation behaviour changes across training distributions.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config
from utils import get_logger, load_flat_ds, load_report, save_report


def main():
    log = get_logger("step_08_bias_variance")
    log.info("Loading MNIST data for classical model bias-variance sweeps...")
    X_train_full, y_train_full, X_test, y_test = load_flat_ds("mnist")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train_full), config.BV_SAMPLE_SIZE, replace=False)
    X_tr = X_train_full[idx]
    y_tr = y_train_full[idx]

    X_tr_norm   = X_tr.astype(np.float32)   / 255.0
    X_test_norm = X_test.astype(np.float32) / 255.0

    results = {}

    # 1. KNN - vary k
    log.info("KNN bias-variance sweep (MNIST subset)...")
    from sklearn.neighbors import KNeighborsClassifier
    knn_train, knn_test = [], []
    for k in config.BV_KNN_K_VALUES:
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        knn_train.append(clf.score(X_tr, y_tr))
        knn_test.append(clf.score(X_test, y_test))
        log.info(f"  k={k:>2}  train={knn_train[-1]:.4f}  test={knn_test[-1]:.4f}")
    results["knn"] = {"k": config.BV_KNN_K_VALUES, "train": knn_train, "test": knn_test}

    # 2. RFC - vary n_estimators
    log.info("RFC bias-variance sweep (MNIST subset)...")
    from sklearn.ensemble import RandomForestClassifier
    rfc_train, rfc_test = [], []
    for n in config.BV_RFC_TREES:
        clf = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=42)
        clf.fit(X_tr, y_tr)
        rfc_train.append(clf.score(X_tr, y_tr))
        rfc_test.append(clf.score(X_test, y_test))
        log.info(f"  n={n:>3}  train={rfc_train[-1]:.4f}  test={rfc_test[-1]:.4f}")
    results["rfc"] = {"n_estimators": config.BV_RFC_TREES, "train": rfc_train, "test": rfc_test}

    # 3. LR - vary C
    log.info("Logistic Regression bias-variance sweep (MNIST subset)...")
    from sklearn.linear_model import LogisticRegression
    lr_train, lr_test = [], []
    for c in config.BV_LR_C_VALUES:
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=c, random_state=42)
        clf.fit(X_tr_norm, y_tr)
        lr_train.append(clf.score(X_tr_norm, y_tr))
        lr_test.append(clf.score(X_test_norm, y_test))
        log.info(f"  C={c:>6}  train={lr_train[-1]:.4f}  test={lr_test[-1]:.4f}")
    results["lr"] = {"C": config.BV_LR_C_VALUES, "train": lr_train, "test": lr_test}

    # 4. CNN epoch history from all datasets
    for ds in config.DATASETS:
        b_path = config.cnn_b_report(ds)
        i_path = config.cnn_i_report(ds)
        if b_path.exists():
            results[f"cnn_baseline_{ds}"] = load_report(b_path).get("history", {})
            log.info(f"CNN baseline history loaded [{ds}].")
        if i_path.exists():
            d = load_report(i_path)
            results[f"cnn_improved_{ds}"]   = d.get("history_improved", {})
            results[f"cnn_no_dropout_{ds}"] = d.get("history_no_dropout", {})
            log.info(f"CNN improved history loaded [{ds}].")

    # Plots
    plt.style.use(config.PLOT_STYLE)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["knn"]["k"], results["knn"]["train"], "o-",
            label="Train accuracy", color="#378ADD")
    ax.plot(results["knn"]["k"], results["knn"]["test"],  "s-",
            label="Test accuracy",  color="#D85A30")
    ax.axvline(x=5, color="gray", linestyle="--", alpha=0.5, label="k=5 (default)")
    ax.set_xlabel("k (number of neighbors)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("KNN: Bias-Variance Tradeoff (MNIST)\nsmall k = overfit, large k = underfit",
                 fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "bias_variance_knn.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["rfc"]["n_estimators"], results["rfc"]["train"], "o-",
            label="Train accuracy", color="#378ADD")
    ax.plot(results["rfc"]["n_estimators"], results["rfc"]["test"],  "s-",
            label="Test accuracy",  color="#D85A30")
    ax.set_xlabel("Number of trees", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("RFC: Effect of Ensemble Size (MNIST)\nmore trees = lower variance", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "bias_variance_rfc.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(results["lr"]["C"], results["lr"]["train"], "o-",
                label="Train accuracy", color="#378ADD")
    ax.semilogx(results["lr"]["C"], results["lr"]["test"],  "s-",
                label="Test accuracy",  color="#D85A30")
    ax.set_xlabel("Regularisation strength C (log scale)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Logistic Regression: Regularisation (MNIST)\nsmall C = underfit, large C = overfit",
                 fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "bias_variance_lr.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # CNN curves across all datasets
    colors_ds = {"mnist": "#3C3489", "emnist": "#D85A30", "usps": "#639922"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ds in config.DATASETS:
        key = f"cnn_improved_{ds}"
        if key in results and results[key].get("accuracy"):
            h = results[key]
            c = colors_ds[ds]
            axes[0].plot(h["accuracy"],     color=c, linestyle="--", alpha=0.6,
                         label=f"{ds} train")
            axes[0].plot(h["val_accuracy"], color=c, linestyle="-",
                         label=f"{ds} val")
            axes[1].plot(h["loss"],         color=c, linestyle="--", alpha=0.6,
                         label=f"{ds} train")
            axes[1].plot(h["val_loss"],     color=c, linestyle="-",
                         label=f"{ds} val")
    axes[0].set_title("CNN Improved - Accuracy Across Datasets", fontsize=11)
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[1].set_title("CNN Improved - Loss Across Datasets", fontsize=11)
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    fig.suptitle("CNN Training Curves: MNIST vs EMNIST vs USPS", fontsize=13)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "bias_variance_cnn_all_datasets.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("CNN multi-dataset training curve plot saved.")

    save_report(results, config.BV_REPORT)
    log.info("Step 08 complete.")


if __name__ == "__main__":
    main()
