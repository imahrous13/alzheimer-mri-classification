from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from .dataset import CLASS_NAMES
from .paths import RAW_DATA_DIR, RESULTS_DIR


def segment_brain_outline(path: Path, size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an illustrative active contour using geometry relative to image size."""
    with Image.open(path) as source:
        image = np.asarray(
            source.convert("L").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    smooth = gaussian(image, sigma=2.0)
    angle = np.linspace(0, 2 * np.pi, 240)
    center = (size - 1) / 2
    initial = np.column_stack(
        [center + 0.39 * size * np.sin(angle), center + 0.39 * size * np.cos(angle)]
    )
    contour = active_contour(
        smooth,
        initial,
        alpha=0.015,
        beta=10,
        gamma=0.001,
        max_num_iter=250,
        boundary_condition="periodic",
    )
    return image, initial, contour


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create active-contour examples for each class.")
    parser.add_argument("--dataset", type=Path, default=RAW_DATA_DIR / "original")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "segmentation" / "active_contours.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure, axes = plt.subplots(2, 2, figsize=(9, 9))
    for axis, label in zip(axes.flat, CLASS_NAMES, strict=False):
        image_path = sorted((args.dataset / label).glob("*"))[0]
        image, initial, contour = segment_brain_outline(image_path)
        axis.imshow(image, cmap="gray")
        axis.plot(initial[:, 1], initial[:, 0], "--", color="#D59A2A", linewidth=1, label="Initial")
        axis.plot(contour[:, 1], contour[:, 0], color="#3569A8", linewidth=1.5, label="Fitted")
        axis.set_title(label)
        axis.axis("off")
    axes.flat[0].legend(loc="lower right", frameon=True)
    figure.suptitle("Active-contour exploration on one original MRI per class")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
