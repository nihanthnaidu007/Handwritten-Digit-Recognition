"""Step 10 - Visualisations: GradCAM, filters, feature maps, t-SNE, misclassifications.

New in v2:
  - GradCAM: gradient-weighted class activation maps showing which pixels
    drove the CNN prediction. Run on all three datasets.
  - All existing visualisations retained and run per-dataset where applicable.
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config
from utils import (get_logger, load_flat_ds, load_raw_ds, get_device,
                   load_cnn_improved, cnn_predict, cnn_extract_features,
                   GradCAM, make_tsne)


def visualise_gradcam(ds, device, log):
    """GradCAM for one example per digit class from the test set."""
    if not config.cnn_imp(ds).exists():
        log.info("  [%s] CNN model not found -- skipping GradCAM.", ds)
        return

    log.info("  [%s] Computing GradCAM...", ds)
    cnn = load_cnn_improved(device, ds)

    # Target: second conv of block 2 = features[10] (Conv2d 64->64)
    target_layer = cnn.features[10]
    cam_engine   = GradCAM(cnn, target_layer)

    X_raw, y_raw, X_test_raw, y_test_raw = load_raw_ds(ds)
    samples = [int(np.where(y_test_raw == d)[0][0]) for d in range(10)]

    fig, axes = plt.subplots(3, 10, figsize=(22, 7))
    for col, idx in enumerate(samples):
        img    = X_test_raw[idx]
        true_d = y_test_raw[idx]

        x_t = torch.tensor(
            img[np.newaxis, np.newaxis].astype(np.float32) / 255.0,
            device=device,
        )
        heatmap, pred_d = cam_engine.compute(x_t)

        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"True: {true_d}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(heatmap, cmap="jet")
        axes[1, col].set_title(
            f"Pred: {pred_d}", fontsize=8,
            color="green" if pred_d == true_d else "red"
        )
        axes[1, col].axis("off")

        img_rgb  = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        heat_rgb = plt.cm.jet(heatmap)[:, :, :3]
        overlay  = np.clip(0.55 * img_rgb + 0.45 * heat_rgb, 0, 1)
        axes[2, col].imshow(overlay)
        axes[2, col].set_title("overlay", fontsize=7)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("GradCAM",  fontsize=9)
    axes[2, 0].set_ylabel("Overlay",  fontsize=9)

    fig.suptitle(
        f"GradCAM -- CNN Improved [{ds.upper()}]\n"
        "Heatmap shows which pixels drove each prediction (red = high importance)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / f"gradcam_{ds}.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    cam_engine.remove()
    log.info("  [%s] GradCAM saved.", ds)


def visualise_filters_and_featuremaps(device, log):
    """CNN filter and feature map visualisation (MNIST model)."""
    if not config.cnn_imp("mnist").exists():
        return
    log.info("  Visualising CNN filters and feature maps (MNIST)...")
    cnn = load_cnn_improved(device, "mnist")

    first_conv = cnn.features[0]
    filters    = first_conv.weight.data.cpu().numpy()   # (32, 1, 3, 3)

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            ax.imshow(filters[i, 0], cmap="RdBu", interpolation="nearest")
            ax.set_title(f"F{i+1}", fontsize=7)
        ax.axis("off")
    fig.suptitle("CNN Learned Conv Layer 1 Filters (32 filters, 3x3)\n"
                 "Network learns edge/stroke detectors automatically", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "cnn_filters_layer1.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # Feature maps
    X_test_raw  = load_raw_ds("mnist")[2]
    y_test_raw  = load_raw_ds("mnist")[3]
    feat_outputs = {}

    def hook_fn(module, inp, out):
        feat_outputs["fmap"] = out.detach().cpu()

    handle  = first_conv.register_forward_hook(hook_fn)
    samples = [int(np.where(y_test_raw == d)[0][0]) for d in range(10)]
    X_s     = torch.tensor(
        X_test_raw[samples][:, np.newaxis].astype(np.float32) / 255.0,
        device=device,
    )
    with torch.no_grad():
        cnn(X_s)
    handle.remove()

    fmaps = feat_outputs["fmap"].numpy()
    fig, axes = plt.subplots(10, 9, figsize=(18, 20))
    for row in range(10):
        axes[row, 0].imshow(X_test_raw[samples[row]], cmap="gray")
        axes[row, 0].set_title(f"Digit {row}", fontsize=8)
        axes[row, 0].axis("off")
        for col in range(8):
            axes[row, col + 1].imshow(fmaps[row, col], cmap="viridis")
            axes[row, col + 1].set_title(f"F{col+1}", fontsize=7)
            axes[row, col + 1].axis("off")
    fig.suptitle("Feature Maps -- Conv Layer 1 (first 8 of 32 filters per digit)\n"
                 "Each filter responds to different stroke patterns", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "cnn_feature_maps.png",
                dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Filter and feature map visualisation saved.")


def visualise_tsne(device, log):
    """t-SNE: raw pixels vs CNN features (MNIST)."""
    log.info("  Computing t-SNE (MNIST, 2000 samples)...")
    X_test_flat = load_flat_ds("mnist")[2]
    y_test_flat = load_flat_ds("mnist")[3]
    rng      = np.random.default_rng(42)
    idx      = rng.choice(len(X_test_flat), 2000, replace=False)
    X_sample = X_test_flat[idx]
    y_sample = y_test_flat[idx]
    cmap     = plt.cm.get_cmap("tab10", 10)

    emb_raw = make_tsne().fit_transform(X_sample)
    log.info("  t-SNE raw done.")

    emb_cnn = None
    if config.cnn_imp("mnist").exists():
        cnn     = load_cnn_improved(device, "mnist")
        feats   = cnn_extract_features(cnn, X_sample, device)
        emb_cnn = make_tsne().fit_transform(feats)
        log.info("  t-SNE CNN features done.")

    if emb_cnn is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, emb, title in [
            (axes[0], emb_raw, "Raw Pixels (784-D) -- overlapping clusters"),
            (axes[1], emb_cnn, "CNN Features (128-D) -- clean separation"),
        ]:
            for d in range(10):
                mask = y_sample == d
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=[cmap(d)], label=str(d), s=8, alpha=0.7)
            ax.set_title(f"t-SNE: {title}", fontsize=11)
            ax.legend(title="Digit", fontsize=8, markerscale=2)
            ax.axis("off")
        fig.suptitle("t-SNE: Raw Pixels vs CNN Learned Representations (MNIST)",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / "tsne_comparison.png",
                    dpi=config.PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info("  t-SNE comparison saved.")


def visualise_misclassifications(ds, device, log):
    """Show first 20 misclassified images for KNN, RFC, CNN on a dataset."""
    log.info("  [%s] Misclassification analysis...", ds)
    X_test_flat = load_flat_ds(ds)[2]
    y_test_flat = load_flat_ds(ds)[3]
    X_test_raw  = load_raw_ds(ds)[2]

    models_to_check = [
        ("KNN",           config.knn_model(ds), False),
        ("Random Forest", config.rfc_model(ds), False),
        ("CNN Improved",  config.cnn_imp(ds),   True),
    ]
    for model_name, model_path, is_cnn in models_to_check:
        if not model_path.exists():
            continue
        if is_cnn:
            cnn   = load_cnn_improved(device, ds)
            preds = cnn_predict(cnn, X_test_flat, device)
        else:
            clf   = joblib.load(model_path)
            X_norm = X_test_flat.astype(np.float32) / 255.0
            preds = clf.predict(X_norm)

        wrong  = np.where(preds != y_test_flat)[0]
        sample = wrong[:20]
        fig, axes = plt.subplots(2, 10, figsize=(20, 5))
        for ax, i in zip(axes.flat, sample):
            ax.imshow(X_test_raw[i], cmap="gray", interpolation="nearest")
            ax.set_title(f"T:{y_test_flat[i]} P:{preds[i]}", fontsize=8, color="red")
            ax.axis("off")
        safe = model_name.lower().replace(" ", "_")
        fig.suptitle(
            f"{model_name} [{ds.upper()}] -- Misclassified  "
            f"({len(wrong)} errors  |  T=True  P=Predicted)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / f"misclassifications_{safe}_{ds}.png",
                    dpi=config.PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info("    %s [%s]: %d errors.", model_name, ds, len(wrong))


def main():
    log    = get_logger("step_10_visualisations")
    device = get_device()
    log.info("Step 10 -- Visualisations (GradCAM, filters, t-SNE, misclassifications)")

    # GradCAM -- all three datasets
    for ds in config.DATASETS:
        visualise_gradcam(ds, device, log)

    # CNN filters + feature maps (MNIST model)
    visualise_filters_and_featuremaps(device, log)

    # t-SNE (MNIST)
    visualise_tsne(device, log)

    # Misclassifications -- all datasets
    for ds in config.DATASETS:
        visualise_misclassifications(ds, device, log)

    log.info("Step 10 complete.")


if __name__ == "__main__":
    main()
