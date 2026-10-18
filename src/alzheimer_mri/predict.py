from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from .model import build_model
from .paths import MODELS_DIR


def prediction_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


def discover_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict dementia class from MRI images.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "best_model.pt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = build_model(len(class_names), dropout=checkpoint.get("dropout", 0.35)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = prediction_transform(checkpoint["image_size"])

    rows = []
    for path in discover_images(args.input):
        with Image.open(path) as source:
            tensor = transform(source.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
        predicted_index = int(probabilities.argmax())
        confidence = float(probabilities[predicted_index])
        row = {
            "path": str(path),
            "predicted_label": class_names[predicted_index],
            "confidence": confidence,
            "review_recommended": confidence < args.uncertainty_threshold,
        }
        row.update(
            {f"probability_{name}": float(probabilities[index]) for index, name in enumerate(class_names)}
        )
        rows.append(row)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
