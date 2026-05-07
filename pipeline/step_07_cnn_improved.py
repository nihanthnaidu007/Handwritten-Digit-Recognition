"""Step 07 — Improved CNN with regularisation + data augmentation on all datasets.

New in v2:
  - Data augmentation on training set (random affine + random erasing) via
    make_augmented_loaders() — augmentation is applied per-batch so the model
    sees slightly different versions of each image every epoch.
  - Same overfitting demo (no-dropout vs dropout) as before, now run per-dataset.
  - Augmentation is clearly labelled in all plot titles so the viewer knows
    this is a separate regularisation technique beyond dropout/L2/early-stopping.
"""

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config
from models import CNNImproved, CNNNoDropout
from utils import (get_logger, load_cnn_ds, make_loaders, make_augmented_loaders,
                   train_epoch, eval_epoch, predict_all, get_device,
                   plot_confusion_matrix, plot_training_curves, save_report)


def run_training(model, trn_loader, val_loader, p, device, patience=None, log=None):
    _log = log.info if log is not None else print
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=p["lr"], weight_decay=p.get("weight_decay", 0)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3,
    )
    history      = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_acc = 0.0
    best_weights = None
    no_improve   = 0

    for epoch in range(1, p["epochs"] + 1):
        trn_loss, trn_acc = train_epoch(model, trn_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history["loss"].append(trn_loss)
        history["accuracy"].append(trn_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        _log(f"  Epoch {epoch:>2}/{p['epochs']}  "
             f"loss={trn_loss:.4f}  acc={trn_acc:.4f}  "
             f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if patience is not None:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    _log(f"  Early stopping at epoch {epoch}.")
                    model.load_state_dict(best_weights)
                    break
    return history


def train_on_dataset(ds: str, device, log):
    log.info(f"\n{'─'*50}\n  Dataset: {ds.upper()}\n{'─'*50}")
    X_train, y_train, X_test, y_test = load_cnn_ds(ds)
    p = config.CNN_IMP_PARAMS
    torch.manual_seed(p["random_state"])

    # Standard loaders (for no-dropout overfitting demo — no augmentation)
    trn_loader_plain, val_loader, tst_loader = make_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=p["batch_size"], val_split=p["val_split"],
    )

    # Augmented loaders (for the improved model)
    trn_loader_aug, _, _ = make_augmented_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=p["batch_size"], val_split=p["val_split"],
    )

    # ── A: No-dropout (overfitting demo) ─────────────────────────────────────
    log.info(f"  [{ds}] Training without dropout (overfitting demo)...")
    model_nd = CNNNoDropout().to(device)
    hist_nd  = run_training(
        model_nd, trn_loader_plain, val_loader,
        {**p, "epochs": 20, "weight_decay": 0},
        device, patience=None, log=log,
    )

    # ── B: Improved CNN — dropout + L2 + early stopping + augmentation ────────
    log.info(f"  [{ds}] Training improved CNN (dropout + L2 + augmentation)...")
    model_imp = CNNImproved(
        dropout_conv=p["dropout_conv"],
        dropout_dense=p["dropout_dense"],
    ).to(device)
    hist_imp = run_training(
        model_imp, trn_loader_aug, val_loader, p,
        device, patience=p["patience"], log=log,
    )

    torch.save(model_imp.state_dict(), config.cnn_imp(ds))
    log.info(f"  Model saved -> {config.cnn_imp(ds).name}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    nd_preds,  nd_trues  = predict_all(model_nd,  tst_loader, device)
    imp_preds, imp_trues = predict_all(model_imp, tst_loader, device)
    nd_acc  = accuracy_score(nd_trues,  nd_preds)
    imp_acc = accuracy_score(imp_trues, imp_preds)
    log.info(f"  [{ds}] No-dropout accuracy : {nd_acc:.4f}")
    log.info(f"  [{ds}] Improved   accuracy : {imp_acc:.4f}")

    cm     = confusion_matrix(imp_trues, imp_preds)
    report = classification_report(imp_trues, imp_preds, output_dict=True)

    # ── Regularisation comparison plot ────────────────────────────────────────
    plt.style.use(config.PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for hist, label, color in [
        (hist_nd,  "Without dropout (overfitting)",             "red"),
        (hist_imp, "With dropout + L2 + augmentation (reg.)",   "green"),
    ]:
        axes[0].plot(hist["accuracy"],     color=color, linestyle="--", alpha=0.7,
                     label=f"{label} — train")
        axes[0].plot(hist["val_accuracy"], color=color, linestyle="-",
                     label=f"{label} — val")
        axes[1].plot(hist["loss"],         color=color, linestyle="--", alpha=0.7,
                     label=f"{label} — train")
        axes[1].plot(hist["val_loss"],     color=color, linestyle="-",
                     label=f"{label} — val")
    axes[0].set_title(f"Accuracy: Regularisation Effect [{ds.upper()}]", fontsize=11)
    axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=7)
    axes[1].set_title(f"Loss: Regularisation Effect [{ds.upper()}]", fontsize=11)
    axes[1].set_xlabel("Epoch"); axes[1].legend(fontsize=7)
    fig.suptitle(
        f"CNN Regularisation Study [{ds.upper()}]\n"
        "Dropout Off vs Dropout + L2 + Data Augmentation", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / f"regularisation_comparison_cnn_{ds}.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    plot_training_curves(
        hist_imp,
        title=f"CNN Improved [{ds.upper()}] — Training & Validation",
        save_path=config.PLOTS_DIR / f"training_curves_cnn_improved_{ds}.png",
    )
    plot_confusion_matrix(
        cm,
        title=f"CNN Improved [{ds.upper()}]  acc={imp_acc*100:.2f}%",
        save_path=config.PLOTS_DIR / f"confusion_matrix_cnn_improved_{ds}.png",
    )
    save_report(
        {
            "model": "CNN Improved",
            "dataset": ds,
            "test_accuracy": imp_acc,
            "no_dropout_test_accuracy": nd_acc,
            "confusion_matrix": cm,
            "classification_report": report,
            "history_improved": hist_imp,
            "history_no_dropout": hist_nd,
            "params": p,
        },
        config.cnn_i_report(ds),
    )


def main():
    log    = get_logger("step_07_cnn_improved")
    device = get_device()
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    for ds in config.DATASETS:
        train_on_dataset(ds, device, log)
    log.info("\nStep 07 complete — Improved CNN trained on all datasets.")


if __name__ == "__main__":
    main()
