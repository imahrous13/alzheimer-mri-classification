from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .dataset import CLASS_NAMES, load_split_rows, make_loader
from .model import build_model
from .paths import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, ensure_output_dirs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(rows: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = rows["label"].value_counts().reindex(CLASS_NAMES).astype(float)
    weights = np.sqrt(counts.sum() / counts.clip(lower=1))
    weights /= weights.mean()
    return torch.tensor(weights.values, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels, _ in tqdm(loader, leave=False, unit="batch"):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())

    labels_array = np.asarray(labels_all)
    predictions_array = np.asarray(predictions_all)
    return {
        "loss": total_loss / len(labels_array),
        "accuracy": float((labels_array == predictions_array).mean()),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels_array, predictions_array)
        ),
        "macro_f1": float(
            f1_score(labels_array, predictions_array, average="macro", zero_division=0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Alzheimer MRI classifier.")
    parser.add_argument(
        "--splits", type=Path, default=PROCESSED_DATA_DIR / "dataset_splits.csv"
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--train-source",
        choices=["all", "original"],
        default="all",
        help="Whether training may use the augmented collection or original images only.",
    )
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    parser.add_argument("--run-dir", type=Path, default=RESULTS_DIR / "training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = load_split_rows(args.splits, "train")
    if args.train_source == "original":
        train_rows = train_rows[train_rows["dataset"] == "original"].copy()
    validation_rows = load_split_rows(args.splits, "validation")
    if args.max_train_samples and len(train_rows) > args.max_train_samples:
        sampled_groups = []
        for _, group in train_rows.groupby("label"):
            sample_size = max(
                1, round(args.max_train_samples * len(group) / len(train_rows))
            )
            sampled_groups.append(
                group.sample(min(sample_size, len(group)), random_state=args.seed)
            )
        train_rows = pd.concat(sampled_groups, ignore_index=True)

    train_loader = make_loader(
        train_rows, args.image_size, args.batch_size, training=True, workers=args.workers
    )
    validation_loader = make_loader(
        validation_rows,
        args.image_size,
        args.batch_size,
        training=False,
        workers=args.workers,
    )

    model = build_model(num_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, device))
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / args.checkpoint_name
    history: list[dict] = []
    best_score = -1.0
    stale_epochs = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device, optimizer=None
        )
        scheduler.step(validation_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row))

        score = validation_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "best_validation_macro_f1": best_score,
                    "training_arguments": vars(args),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    metadata = {
        "device": str(device),
        "duration_seconds": round(time.time() - started, 2),
        "best_validation_macro_f1": best_score,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_source": args.train_source,
        "checkpoint": checkpoint_path.relative_to(checkpoint_path.parents[1]).as_posix(),
        "arguments": vars(args),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
