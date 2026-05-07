"""Step 11 - Interactive Gradio Demo with GradCAM explainability.

New in v2:
  - Dataset selector: user picks MNIST / EMNIST / USPS model.
  - GradCAM heatmap output panel -- shows which parts of the drawn digit
    drove the CNN's prediction (displayed as a colour overlay).
  - USPS-specific preprocessing: canvas drawing is downsampled to 16x16
    then upscaled to 28x28 to match the USPS training distribution.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config
from utils import get_logger, get_device, load_cnn_improved, GradCAM

_models  = {}   # ds -> model
_cams    = {}   # ds -> GradCAM engine
_device  = None


def _get_model_and_cam(ds):
    global _device
    if _device is None:
        _device = get_device()
    if ds not in _models:
        model = load_cnn_improved(_device, ds)
        _models[ds] = model
        _cams[ds]   = GradCAM(model, model.features[10])
    return _models[ds], _cams[ds], _device


def extract_gray(image):
    """Convert Gradio sketchpad output to (H,W) float32 with ink=bright."""
    if image is None:
        return None
    if isinstance(image, dict):
        arr = image.get("composite")
        if arr is None:
            layers = image.get("layers", [])
            arr = layers[0] if layers else None
        if arr is None:
            return None
    else:
        arr = image
    arr = np.array(arr, dtype=np.uint8)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgb  = arr[:, :, :3].astype(np.float32)
        gray = 255.0 - np.mean(rgb, axis=2)
    elif arr.ndim == 3:
        rgb  = arr.astype(np.float32)
        gray = 255.0 - np.mean(rgb, axis=2)
    else:
        return None
    return gray


def preprocess_for_dataset(gray_arr, ds):
    """
    Resize the (H,W) float32 canvas to 28x28, applying dataset-specific
    preprocessing to match each model's training distribution.

    MNIST / EMNIST: direct LANCZOS resize to 28x28.
    USPS:           downsample to 16x16 first (simulating postal scan
                    resolution), then upscale to 28x28 with LANCZOS.
                    This matches exactly how USPS training images were prepared
                    in step_01, so the model sees familiar-looking input.
    """
    from PIL import Image as PILImage

    img_pil = PILImage.fromarray(np.clip(gray_arr, 0, 255).astype(np.uint8))

    if ds == "usps":
        # Step 1: simulate postal scanner low resolution
        img_pil = img_pil.resize((16, 16), PILImage.LANCZOS)
        # Step 2: upscale back to 28x28 -- exactly what step_01 does
        img_pil = img_pil.resize((28, 28), PILImage.LANCZOS)
    else:
        img_pil = img_pil.resize((28, 28), PILImage.LANCZOS)

    return np.array(img_pil).astype(np.float32) / 255.0


def predict_digit(image, ds):
    """Run CNN inference + GradCAM on the drawn digit."""
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    gray = extract_gray(image)
    if gray is None or gray.max() < 5:
        blank_probs = {str(i): 0.1 for i in range(10)}
        return blank_probs, None

    # Dataset-specific preprocessing
    img_arr = preprocess_for_dataset(gray, ds)

    tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)   # (1,1,28,28)

    model, cam_engine, device = _get_model_and_cam(ds)
    tensor_dev = tensor.to(device)

    # Probabilities
    with torch.no_grad():
        out   = model(tensor_dev)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]

    label_dict = {str(i): float(probs[i]) for i in range(10)}

    # GradCAM (requires grad)
    try:
        heatmap, pred_class = cam_engine.compute(tensor_dev)

        img_rgb  = np.stack([img_arr, img_arr, img_arr], axis=-1)
        heat_rgb = plt.cm.jet(heatmap)[:, :, :3]
        overlay  = np.clip(0.55 * img_rgb + 0.45 * heat_rgb, 0, 1)

        overlay_pil = PILImage.fromarray((overlay * 255).astype(np.uint8))
        overlay_pil = overlay_pil.resize((280, 280), PILImage.NEAREST)
        overlay_out = np.array(overlay_pil)
    except Exception:
        overlay_out = None

    return label_dict, overlay_out


def main():
    log = get_logger("step_11_demo")

    if not config.cnn_imp("mnist").exists():
        log.info("CNN improved model (MNIST) not found. Run step 7 first.")
        return

    try:
        import gradio as gr
    except ImportError:
        log.info("Gradio not installed. Run:  pip install gradio")
        return

    import torch
    device = get_device()
    log.info("Device: %s", device)
    if torch.cuda.is_available():
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    available_ds = [ds for ds in config.DATASETS if config.cnn_imp(ds).exists()]
    for ds in available_ds:
        _get_model_and_cam(ds)
        log.info("Model loaded: %s", ds)

    log.info("Launching demo...")

    with gr.Blocks(title="MNIST Digit Recogniser + GradCAM", theme=gr.themes.Default()) as demo:
        gr.Markdown(
            """
            # Handwritten Digit Recogniser -- GradCAM Edition
            **AI 681 Machine Learning | LIU**

            Draw a digit (0-9), select a model, and click **Predict**.
            The **GradCAM heatmap** shows *which pixels* drove the CNN's decision
            (red = high importance, blue = low importance).
            """
        )

        with gr.Row():
            ds_selector = gr.Radio(
                choices=available_ds,
                value=available_ds[0],
                label="Model trained on dataset",
            )

        with gr.Row():
            with gr.Column(scale=1):
                canvas = gr.Sketchpad(
                    label="Draw your digit here",
                    canvas_size=(280, 280),
                    type="numpy",
                )
                with gr.Row():
                    btn_pred  = gr.Button("Predict", variant="primary",    scale=2)
                    btn_clear = gr.Button("Clear",   variant="secondary",  scale=1)

            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="CNN Prediction (confidence per class)",
                    num_top_classes=10,
                )
                output_cam = gr.Image(
                    label="GradCAM Heatmap (what the CNN focused on)",
                    type="numpy",
                )

        btn_pred.click(
            fn=predict_digit,
            inputs=[canvas, ds_selector],
            outputs=[output_label, output_cam],
        )
        btn_clear.click(
            fn=lambda: (None, {str(i): 0.0 for i in range(10)}, None),
            inputs=None,
            outputs=[canvas, output_label, output_cam],
        )

        gr.Markdown(
            """
            ---
            **How it works:** Your drawing -> resized to 28x28 -> normalised -> 2 conv blocks
            (BatchNorm + Dropout + Augmentation-trained) -> Dense(256) -> Dense(128) -> Softmax(10).

            **GradCAM** backpropagates the predicted class score through the last conv layer,
            weights feature maps by gradient importance, and overlays the result on your digit.

            **Datasets & preprocessing:**
            - **MNIST** -- 60k training images (the original benchmark). Direct 28x28 resize.
            - **EMNIST** -- 240k images (harder, more varied handwriting styles). Direct 28x28 resize.
            - **USPS** -- 7k images from real postal mail scanners (true out-of-domain transfer).
              Your drawing is first downsampled to 16x16 (simulating the scanner's low resolution),
              then upsampled back to 28x28 -- exactly matching how USPS training data was prepared.
              Lower confidence on USPS is expected: only ~730 training examples per digit class.

            *Achieved accuracy: MNIST ~99% | EMNIST ~99% | USPS ~90% (small dataset + postal domain)*
            """
        )

    log.info("Demo running at http://127.0.0.1:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
