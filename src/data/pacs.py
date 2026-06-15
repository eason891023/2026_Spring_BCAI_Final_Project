import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PACS_URL = (
    "https://huggingface.co/datasets/flwrlabs/pacs/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
NUM_CLASSES = 7
IMAGE_SIZE = 64


class _PACSDomainDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image_obj = row["image"]
        if isinstance(image_obj, dict):
            if image_obj.get("bytes") is not None:
                image = Image.open(io.BytesIO(image_obj["bytes"]))
            else:
                image = Image.open(image_obj["path"])
        else:
            image = image_obj
        image = image.convert("RGB")
        return self.transform(image), int(row["label"])


def _download_pacs(data_dir):
    pacs_dir = Path(data_dir) / "PACS"
    pacs_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = pacs_dir / "train-00000-of-00001.parquet"
    if parquet_path.exists():
        return parquet_path

    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    print(f"[INFO] Downloading PACS from Hugging Face to {parquet_path}")
    with requests.get(PACS_URL, stream=True, timeout=30) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, parquet_path)
    return parquet_path


def _read_pacs_frame(data_dir):
    parquet_path = _download_pacs(data_dir)
    try:
        return pd.read_parquet(parquet_path)
    except ImportError as exc:
        raise ImportError(
            "PACS uses a parquet file. Please install pyarrow, e.g. "
            "`python3 -m pip install pyarrow`."
        ) from exc


def _stratified_split(frame, seed, train_fraction):
    rng = np.random.RandomState(seed)
    train_indices, test_indices = [], []
    for label in sorted(frame["label"].unique()):
        label_indices = frame.index[frame["label"] == label].to_numpy().copy()
        rng.shuffle(label_indices)
        split = int(round(len(label_indices) * train_fraction))
        split = min(max(split, 1), len(label_indices) - 1)
        train_indices.extend(label_indices[:split])
        test_indices.extend(label_indices[split:])
    return frame.loc[train_indices], frame.loc[test_indices]


def get_pacs(batch_size=64, data_dir="./data", seed=42, train_fraction=0.8):
    """
    PACS Domain-IL: each visual domain is one task, label space is shared.

    Domains are ordered as art_painting -> cartoon -> photo -> sketch. The HF
    split contains all images, so each domain is deterministically split into
    stratified train/test subsets per seed.
    """
    frame = _read_pacs_frame(data_dir)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = train_transform

    tasks_train, tasks_test = [], []
    for domain_idx, domain in enumerate(DOMAINS):
        domain_frame = frame[frame["domain"] == domain].reset_index(drop=True)
        train_frame, test_frame = _stratified_split(
            domain_frame, seed=seed + domain_idx * 997, train_fraction=train_fraction)

        generator = torch.Generator()
        generator.manual_seed(seed + domain_idx * 997)
        tasks_train.append(DataLoader(
            _PACSDomainDataset(train_frame, train_transform),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        ))
        tasks_test.append(DataLoader(
            _PACSDomainDataset(test_frame, test_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        ))

    info = {
        "input_size": 3 * IMAGE_SIZE * IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "domains": DOMAINS,
        "image_size": IMAGE_SIZE,
    }
    return tasks_train, tasks_test, info
