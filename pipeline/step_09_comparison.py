"""Step 09 - Cross-dataset comparison, weighted soft ensemble, transfer matrix.

New in v2:
  1. Per-dataset accuracy bar charts (MNIST / EMNIST / USPS side by side).
  2. Cross-dataset transfer accuracy matrix: train on A, test on B.
  3. Weighted soft ensemble -- accuracy-weighted probability averaging.
  4. Cost-sensitive classification and CNN feature extraction retained.
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config
from utils import (get_logger, load_flat_ds, load_report, get_device,
                   plot_confusion_matrix, save_report,
                   load_cnn_improved, cnn_predict, cnn_predict_proba,
                   cnn_extract_features)

MODEL_KEYS = ["lr", "knn", "rfc", "svm", "cnn_baseline", "cnn_improved"]
MODEL_LABELS = {
    "lr":           "Logistic Regression",
    "knn":          "KNN",
    "rfc":          "Random Forest",
    "svm":          "SVM",
    "cnn_baseline": "CNN Baseline",
    "cnn_improved": "CNN Improved",
}
COLORS = ["#7F77DD", "#378ADD", "#639922", "#1D9E75", "#534AB7", "#3C3489"]


def load_reports_for_ds(ds):
    r = {}
    for key, report_fn in [
        ("lr",           config.lr_report),
        ("knn",          config.knn_report),
        ("rfc",          config.rfc_report),
        ("svm",          config.svm_report),
        ("cnn_baseline", config.cnn_b_report),
        ("cnn_improved", config.cnn_i_report),
    ]:
        p = report_fn(ds)
        if p.exists():
            r[key] = load_report(p)
    return r


def weighted_soft_ensemble(ds, device, log):
    """Accuracy-weighted soft-voting ensemble across all available models."""
    from sklearn.metrics import accuracy_score
    import torch.nn.functional as F_nn
    import torch

    X_train, y_train, X_test, y_test = load_flat_ds(ds)
    X_test_norm = X_test.astype(np.float32) / 255.0

    reports = load_reports_for_ds(ds)
    prob_list   = []
    weight_list = []

    # LR
    if config.lr_model(ds).exists() and "lr" in reports:
        clf = joblib.load(config.lr_model(ds))
        prob_list.append(clf.predict_proba(X_test_norm))
        weight_list.append(reports["lr"]["test_accuracy"])
        log.info("  [%s] Ensemble: LR loaded (acc=%.4f)", ds, reports["lr"]["test_accuracy"])

    # KNN
    if config.knn_model(ds).exists() and "knn" in reports:
        clf = joblib.load(config.knn_model(ds))
        prob_list.append(clf.predict_proba(X_test_norm))
        weight_list.append(reports["knn"]["test_accuracy"])
        log.info("  [%s] Ensemble: KNN loaded (acc=%.4f)", ds, reports["knn"]["test_accuracy"])

    # RFC
    if config.rfc_model(ds).exists() and "rfc" in reports:
        clf = joblib.load(config.rfc_model(ds))
        prob_list.append(clf.predict_proba(X_test_norm))
        weight_list.append(reports["rfc"]["test_accuracy"])
        log.info("  [%s] Ensemble: RFC loaded (acc=%.4f)", ds, reports["rfc"]["test_accuracy"])

    # SVM -- no predict_proba; use decision_function + softmax
    if config.svm_model(ds).exists() and "svm" in reports:
        clf   = joblib.load(config.svm_model(ds))
        df    = clf.decision_function(X_test_norm)
        df_t  = torch.tensor(df, dtype=torch.float32)
        probs = F_nn.softmax(df_t, dim=1).numpy()
        prob_list.append(probs)
        weight_list.append(reports["svm"]["test_accuracy"])
        log.info("  [%s] Ensemble: SVM loaded (acc=%.4f)", ds, reports["svm"]["test_accuracy"])

    # CNN Improved
    if config.cnn_imp(ds).exists() and "cnn_improved" in reports:
        cnn   = load_cnn_improved(device, ds)
        probs = cnn_predict_proba(cnn, X_test, device)
        prob_list.append(probs)
        weight_list.append(reports["cnn_improved"]["test_accuracy"])
        log.info("  [%s] Ensemble: CNN loaded (acc=%.4f)", ds, reports["cnn_improved"]["test_accuracy"])

    if len(prob_list) < 2:
        log.info("  [%s] Not enough models for ensemble -- skipping.", ds)
        return None

    weights  = np.array(weight_list)
    weights  = weights / weights.sum()
    stacked  = np.stack(prob_list, axis=0)          # (n_models, N, 10)
    weighted = np.tensordot(weights, stacked, axes=([0], [0]))  # (N, 10)
    ens_pred = np.argmax(weighted, axis=1)
    ens_acc  = accuracy_score(y_test, ens_pred)
    log.info("  [%s] Weighted ensemble accuracy: %.4f", ds, ens_acc)
    return ens_acc


def build_transfer_matrix(device, log):
    """3x3 cross-dataset transfer: row=train dataset, col=test dataset."""
    from sklearn.metrics import accuracy_score
    ds_list = config.DATASETS
    matrix  = np.full((len(ds_list), len(ds_list)), np.nan)

    for i, train_ds in enumerate(ds_list):
        if not config.cnn_imp(train_ds).exists():
            log.info("  Transfer: CNN [%s] not found -- skipping row.", train_ds)
            continue
        cnn = load_cnn_improved(device, train_ds)
        for j, test_ds in enumerate(ds_list):
            _, _, X_test, y_test = load_flat_ds(test_ds)
            preds       = cnn_predict(cnn, X_test, device)
            acc         = accuracy_score(y_test, preds)
            matrix[i, j] = acc
            log.info("  Transfer [%s -> %s]: %.4f", train_ds, test_ds, acc)

    return matrix


def main():
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.linear_model import LogisticRegression

    log    = get_logger("step_09_comparison")
    device = get_device()
    log.info("Step 09 -- Cross-dataset comparison, weighted ensemble, transfer matrix")

    plt.style.use(config.PLOT_STYLE)

    # 1. Per-dataset accuracy bar chart
    ds_accs = {ds: {} for ds in config.DATASETS}
    for ds in config.DATASETS:
        reports = load_reports_for_ds(ds)
        for key in MODEL_KEYS:
            if key in reports:
                ds_accs[ds][key] = reports[key]["test_accuracy"]

    x = np.arange(len(MODEL_KEYS))
    width = 0.25
    ds_colors = {"mnist": "#3C3489", "emnist": "#D85A30", "usps": "#639922"}
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, ds in enumerate(config.DATASETS):
        vals = [ds_accs[ds].get(k, 0) * 100 for k in MODEL_KEYS]
        bars = ax.bar(x + i * width, vals, width, label=ds.upper(),
                      color=ds_colors[ds], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x + width)
    ax.set_xticklabels([MODEL_LABELS[k] for k in MODEL_KEYS], fontsize=9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("All Models x All Datasets -- Test Accuracy", fontsize=13)
    ax.legend(title="Dataset", fontsize=10)
    ax.set_ylim(70, 101)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "comparison_accuracy_all_models_datasets.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Per-dataset accuracy bar chart saved.")

    # 2. Cross-dataset CNN transfer matrix
    log.info("Building cross-dataset transfer matrix...")
    transfer_matrix = build_transfer_matrix(device, log)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(transfer_matrix, cmap="YlGn", vmin=0.6, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Test Accuracy")
    ax.set_xticks(range(len(config.DATASETS)))
    ax.set_yticks(range(len(config.DATASETS)))
    ax.set_xticklabels([d.upper() for d in config.DATASETS], fontsize=11)
    ax.set_yticklabels([d.upper() for d in config.DATASETS], fontsize=11)
    ax.set_xlabel("Test Dataset",  fontsize=12)
    ax.set_ylabel("Train Dataset", fontsize=12)
    ax.set_title("CNN Cross-Dataset Transfer Matrix\n"
                 "How well does each model generalise beyond its training distribution?",
                 fontsize=11)
    for i in range(len(config.DATASETS)):
        for j in range(len(config.DATASETS)):
            val = transfer_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=12, color="black" if val > 0.75 else "white",
                        fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "cross_dataset_transfer_matrix.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Cross-dataset transfer matrix saved.")

    # 3. Weighted soft ensemble per dataset
    log.info("Building weighted soft ensemble...")
    ensemble_accs = {}
    for ds in config.DATASETS:
        acc = weighted_soft_ensemble(ds, device, log)
        if acc is not None:
            ensemble_accs[ds] = acc

    # 4. Per-class F1 comparison (MNIST)
    digits  = [str(d) for d in range(10)]
    x2      = np.arange(len(digits))
    w       = 0.13
    reports = load_reports_for_ds("mnist")
    present = [k for k in MODEL_KEYS if k in reports]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (key, color) in enumerate(zip(present, COLORS)):
        if "classification_report" in reports[key]:
            f1s = [reports[key]["classification_report"].get(d, {}).get("f1-score", 0)
                   for d in digits]
            ax.bar(x2 + i * w, f1s, w, label=MODEL_LABELS[key], color=color, alpha=0.85)
    ax.set_xlabel("Digit class", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Score -- All Models (MNIST)", fontsize=13)
    ax.set_xticks(x2 + w * (len(present) - 1) / 2)
    ax.set_xticklabels(digits)
    ax.legend(fontsize=8)
    ax.set_ylim(0.75, 1.01)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "comparison_f1_per_class_mnist.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Per-class F1 chart saved.")

    # 5. Combined confusion matrices (MNIST)
    reports_mnist = load_reports_for_ds("mnist")
    cms = {MODEL_LABELS[k]: np.array(reports_mnist[k]["confusion_matrix"])
           for k in MODEL_KEYS
           if k in reports_mnist and "confusion_matrix" in reports_mnist[k]}
    n = len(cms)
    if n:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, (name, cm) in zip(axes.flat, cms.items()):
            im = ax.matshow(cm, cmap="viridis")
            key_match = [k for k, v in MODEL_LABELS.items() if v == name]
            acc_val = reports_mnist[key_match[0]]["test_accuracy"] if key_match else 0
            ax.set_title(f"{name}\n{acc_val*100:.2f}%", fontsize=10)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("True", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle("Confusion Matrices -- All Models (MNIST)", fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / "comparison_confusion_matrices_mnist.png",
                    dpi=config.PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info("Combined confusion matrices saved.")

    # 6. Cost-sensitive classification (MNIST)
    cost_acc = None
    if config.cnn_imp("mnist").exists():
        log.info("Cost-sensitive classification (MNIST)...")
        cnn   = load_cnn_improved(device, "mnist")
        X_train, y_train, X_test, y_test = load_flat_ds("mnist")
        probs = cnn_predict_proba(cnn, X_test, device)
        cost_matrix = np.ones((10, 10)) - np.eye(10)
        for a, b in [(1, 7), (7, 1), (3, 8), (8, 3), (4, 9), (9, 4)]:
            cost_matrix[a, b] = 2.0
        cost_pred = np.argmin(probs @ cost_matrix, axis=1)
        std_pred  = np.argmax(probs, axis=1)
        cost_acc  = accuracy_score(y_test, cost_pred)
        std_acc   = accuracy_score(y_test, std_pred)
        log.info("  Standard accuracy : %.4f", std_acc)
        log.info("  Cost-sensitive acc: %.4f", cost_acc)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.matshow(cost_matrix, cmap="Reds")
        plt.colorbar(im, ax=ax, label="Misclassification cost")
        ax.set_title("Cost Matrix (Decision Theory)\n1<->7, 3<->8, 4<->9 cost 2x more",
                     fontsize=11)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / "cost_matrix_decision_theory.png",
                    dpi=config.PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

    # 7. CNN feature extraction (MNIST)
    cnn_feat_results = {}
    if config.cnn_imp("mnist").exists():
        log.info("CNN feature extraction (MNIST)...")
        cnn         = load_cnn_improved(device, "mnist")
        X_train, y_train, X_test, y_test = load_flat_ds("mnist")
        train_feats = cnn_extract_features(cnn, X_train, device)
        test_feats  = cnn_extract_features(cnn, X_test,  device)

        lr_raw = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        lr_raw.fit(X_train[:10000].astype(np.float32) / 255.0, y_train[:10000])
        acc_raw  = lr_raw.score(X_test.astype(np.float32) / 255.0, y_test)

        lr_feat = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        lr_feat.fit(train_feats, y_train)
        acc_feat = lr_feat.score(test_feats, y_test)

        log.info("  LR on raw pixels   : %.4f", acc_raw)
        log.info("  LR on CNN features : %.4f", acc_feat)
        cnn_feat_results = {"lr_raw_pixels": acc_raw, "lr_cnn_features": acc_feat}

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["LR on raw pixels", "LR on CNN features"],
               [acc_raw * 100, acc_feat * 100], color=["#B5D4F4", "#7F77DD"])
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("CNN as Feature Extractor (MNIST)\nRepresentation Learning", fontsize=11)
        ax.set_ylim(85, 100)
        for i, v in enumerate([acc_raw, acc_feat]):
            ax.text(i, v * 100 + 0.1, f"{v*100:.2f}%", ha="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / "cnn_feature_extraction_vs_raw.png",
                    dpi=config.PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

    save_report(
        {
            "per_dataset_accuracies":  ds_accs,
            "transfer_matrix":         transfer_matrix,
            "transfer_datasets":       config.DATASETS,
            "weighted_ensemble":       ensemble_accs,
            "cost_sensitive_accuracy": cost_acc,
            "cnn_feature_extraction":  cnn_feat_results,
        },
        config.CMP_REPORT,
    )
    log.info("Step 09 complete.")


if __name__ == "__main__":
    main()
