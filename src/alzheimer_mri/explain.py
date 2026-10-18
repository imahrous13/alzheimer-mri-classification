from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .model import build_model
from .paths import MODELS_DIR, RESULTS_DIR
from .predict import prediction_transform


def grad_cam(model, inputs: torch.Tensor, class_index: int) -> np.ndarray:
    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Use the final convolution, before batch normalization and pooling, so the
    # activation map retains the most useful spatial evidence for Grad-CAM.
    target_layer = model.features[-1][3]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        logits = model(inputs)
        model.zero_grad(set_to_none=True)
        logits[:, class_index].sum().backward()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * activations).sum(dim=1).relu()
        heatmap = torch.nn.functional.interpolate(
            heatmap.unsqueeze(1), size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        heatmap -= heatmap.min()
        heatmap /= heatmap.max().clamp(min=1e-8)
        return heatmap.detach().cpu().numpy()
    finally:
        forward_handle.remove()
        backward_handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Grad-CAM explanation.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "best_model.pt")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "explanations" / "grad_cam.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(len(checkpoint["class_names"]), checkpoint.get("dropout", 0.35)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    tensor = prediction_transform(checkpoint["image_size"])(image).unsqueeze(0).to(device)
    with torch.no_grad():
        class_index = int(model(tensor).argmax(dim=1).item())
    heatmap = grad_cam(model, tensor, class_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(image.resize((checkpoint["image_size"], checkpoint["image_size"])), cmap="gray")
    ax.imshow(heatmap, cmap="magma", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(f"Grad-CAM: {checkpoint['class_names'][class_index]}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
