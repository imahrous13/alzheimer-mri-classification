from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .paths import RAW_DATA_DIR

CLASS_NAMES = (
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


class MRIDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, image_size: int, training: bool = False):
        self.rows = rows.reset_index(drop=True)
        steps: list = [transforms.Resize((image_size, image_size))]
        if training:
            steps.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(
                        degrees=7,
                        translate=(0.03, 0.03),
                        scale=(0.97, 1.03),
                    ),
                ]
            )
        steps.extend(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = RAW_DATA_DIR / row["path"]
        with Image.open(path) as source:
            image = source.convert("RGB")
        tensor = self.transform(image)
        target = CLASS_TO_INDEX[row["label"]]
        return tensor, target, row["path"]


def load_split_rows(split_csv: Path, split: str) -> pd.DataFrame:
    rows = pd.read_csv(split_csv)
    selected = rows[rows["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split={split!r} in {split_csv}")
    return selected


def make_loader(
    rows: pd.DataFrame,
    image_size: int,
    batch_size: int,
    training: bool,
    workers: int,
) -> DataLoader:
    dataset = MRIDataset(rows, image_size=image_size, training=training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
