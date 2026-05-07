"""Step 06 - CNN Baseline in PyTorch on MNIST, EMNIST, and USPS."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config
from models import CNNBaseline
from utils import (get_logger, load_cnn_ds, make_loaders, train_epoch,
                   eval_epoch, predict_all, get_device,
                   plot_confusion_matrix, plot_training_curves, save_report)


def train_on_dataset(ds, device, log):
    log.info("\n--- Dataset: %s ---", ds.upper())
    X_train, y_train, X_test, y_test = load_cnn_ds(ds)

    p = config.CNN_BASE_PARAMS
    torch.manual_seed(p["random_state"])

    trn_loader, val_loader, tst_loader = make_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=p["batch_size"], val_split=0.2,
    )

    model     = CNNBaseline().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    log.info("  Parameters: %d", n_params)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(1, p["epochs"] + 1):
        trn_loss, trn_acc = train_epoch(model, trn_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history["loss"].append(trn_loss)
        history["accuracy"].append(trn_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        log.info("  [%s] Epoch %2d/%d  loss=%.4f  acc=%.4f  val_loss=%.4f  val_acc=%.4f",
                 ds, epoch, p["epochs"], trn_loss, trn_acc, val_loss, val_acc)

    torch.save(model.state_dict(), config.cnn_base(ds))
    log.info("  Model saved -> %s", config.cnn_base(ds).name)

    y_pred, y_true = predict_all(model, tst_loader, device)
    test_acc = accuracy_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(y_true, y_pred, output_dict=True)
    log.info("  Test accuracy [%s]: %.4f", ds, test_acc)

    plot_training_curves(
        history,
        title=f"CNN Baseline [{ds.upper()}] - Training & Validation",
        save_path=config.PLOTS_DIR / f"training_curves_cnn_baseline_{ds}.png",
    )
    plot_confusion_matrix(
        cm,
        title=f"CNN Baseline [{ds.upper()}]  acc={test_acc*100:.2f}%",
        save_path=config.PLOTS_DIR / f"confusion_matrix_cnn_baseline_{ds}.png",
    )
    save_report(
        {
            "model": "CNN Baseline",
            "dataset": ds,
            "test_accuracy": test_acc,
            "confusion_matrix": cm,
            "classification_report": report,
            "history": history,
            "params": p,
        },
        config.cnn_b_report(ds),
    )


def main():
    log    = get_logger("step_06_cnn_baseline")
    device = get_device()
    log.info("Device: %s", device)
    if torch.cuda.is_available():
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    for ds in config.DATASETS:
        train_on_dataset(ds, device, log)
    log.info("\nStep 06 complete -- CNN Baseline trained on all datasets.")


if __name__ == "__main__":
    main()
