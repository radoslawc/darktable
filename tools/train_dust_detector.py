#!/usr/bin/env python3
"""
Train the Spotless Film dust detector from clean film scans.

This is a script version of the training flow from main.ipynb. It creates
synthetic dust masks on the fly, applies them to grayscale film images, trains
the same 1-channel U-Net used by src/image_processing.py, and saves plain
PyTorch state_dict weights that ImageProcessingService.load_model() can load.

The optional ONNX export is included for future non-Python hosts such as a
darktable module.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def generate_dust_mask(
    image_shape: Tuple[int, int] = (1024, 1024),
    num_blobs: int = 250,
    num_scratches: int = 4,
    num_hairs: int = 50,
    max_blob_size: int = 1,
    max_scratch_length: int = 240,
    max_hair_length: int = 20,
    squiggliness: float = 1.0,
    scale_factor: int = 4,
) -> np.ndarray:
    """Generate a soft float32 dust mask in [0, 1]."""
    height, width = image_shape
    hi_height, hi_width = height * scale_factor, width * scale_factor
    mask_hi = np.zeros((hi_height, hi_width), dtype=np.uint8)
    temp = np.zeros_like(mask_hi)

    def jitter_count(base: int) -> int:
        return random.randint(max(0, int(base * 0.8)), max(0, int(base * 1.2)))

    num_blobs = jitter_count(num_blobs)
    num_scratches = jitter_count(num_scratches)
    num_hairs = jitter_count(num_hairs)
    max_blob_size = max(1, max_blob_size + random.randint(-1, 2))

    for _ in range(num_blobs):
        temp.fill(0)
        center = (random.randint(0, hi_width - 1), random.randint(0, hi_height - 1))
        axes = (
            random.randint(max(1, max_blob_size - 1), max_blob_size + 1) * scale_factor,
            random.randint(1, max(1, max_blob_size // 2) + 1) * scale_factor,
        )
        angle = random.randint(0, 180)
        cv2.ellipse(temp, center, axes, angle, 0, 360, 255, -1)
        mask_hi = cv2.addWeighted(mask_hi, 1.0, temp, random.uniform(0.3, 1.0), 0)

    for _ in range(num_scratches):
        temp.fill(0)
        x1, y1 = random.randint(0, hi_width - 1), random.randint(0, hi_height - 1)
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(
            int(max_scratch_length * 0.7), int(max_scratch_length * 1.3)
        ) * scale_factor
        x2 = int(x1 + length * math.cos(angle))
        y2 = int(y1 + length * math.sin(angle))
        cv2.line(temp, (x1, y1), (x2, y2), 255, scale_factor, lineType=cv2.LINE_AA)
        mask_hi = cv2.addWeighted(mask_hi, 1.0, temp, random.uniform(0.0, 1.0), 0)

    for _ in range(num_hairs):
        temp.fill(0)
        x, y = random.randint(0, hi_width - 1), random.randint(0, hi_height - 1)
        hair_length = random.randint(
            int(max_hair_length * 0.8), int(max_hair_length * 1.2)
        ) * scale_factor
        num_segments = max(3, hair_length // random.randint(8, 12))
        hair_squiggliness = squiggliness * random.uniform(0.8, 1.2)
        angle = random.uniform(0, 2 * math.pi)
        dx_base, dy_base = math.cos(angle), math.sin(angle)
        points = []

        for _ in range(num_segments):
            segment_length = random.uniform(8, 12) * scale_factor
            dx = dx_base * segment_length + hair_squiggliness * random.uniform(
                -segment_length, segment_length
            )
            dy = dy_base * segment_length + hair_squiggliness * random.uniform(
                -segment_length, segment_length
            )
            x = int(np.clip(x + dx, 0, hi_width - 1))
            y = int(np.clip(y + dy, 0, hi_height - 1))
            points.append((x, y))

        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32)
            thickness = random.randint(1, 2) * scale_factor
            cv2.polylines(temp, [pts], False, 255, thickness, lineType=cv2.LINE_AA)
            mask_hi = cv2.addWeighted(mask_hi, 1.0, temp, random.uniform(0.0, 1.0), 0)

    mask_lo = cv2.resize(mask_hi, (width, height), interpolation=cv2.INTER_AREA)
    return mask_lo.astype(np.float32) / 255.0


def apply_dust_to_grayscale_image(
    image_gray: np.ndarray, mask: np.ndarray, intensity: int
) -> np.ndarray:
    image_float = image_gray.astype(np.float32)
    image_float += mask.astype(np.float32) * float(intensity)
    return np.clip(image_float, 0, 255).astype(np.uint8)


class DustDataset(Dataset):
    """Clean image dataset with synthetic dust applied on each sample."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        image_size: int,
        augment: bool = True,
        mask_scale_factor: int = 4,
    ) -> None:
        self.image_paths = list(image_paths)
        self.image_size = image_size
        self.augment = augment
        self.mask_scale_factor = mask_scale_factor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self._load_grayscale(self.image_paths[idx])

        if self.augment:
            clean = self._augment_clean_image(clean)

        mask = generate_dust_mask(
            image_shape=(self.image_size, self.image_size),
            squiggliness=random.uniform(0.5, 1.5),
            scale_factor=self.mask_scale_factor,
        )
        polarity = random.choice([-1, 1])
        dusty = apply_dust_to_grayscale_image(
            clean, mask, intensity=polarity * random.randint(80, 255)
        )

        if self.augment and random.random() < 0.35:
            dusty = self._adjust_contrast(dusty)

        image_tensor = torch.from_numpy(dusty.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy((mask > 0.05).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor

    def _load_grayscale(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            return np.array(image, dtype=np.uint8)

    def _augment_clean_image(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            image = np.fliplr(image)
        if random.random() < 0.3:
            image = np.flipud(image)
        if random.random() < 0.25:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k)
        return np.ascontiguousarray(image)

    @staticmethod
    def _adjust_contrast(image: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.85, 1.2)
        beta = random.uniform(-10, 10)
        return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


class UNet(nn.Module):
    """Exact layer names and shapes used by src/image_processing.py."""

    def __init__(self) -> None:
        super().__init__()

        def conv_block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(),
            )

        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.middle = conv_block(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        middle = self.middle(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(middle), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class UNetWithSigmoid(nn.Module):
    def __init__(self, model: UNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def dice_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target.sum()
    negative = target.numel() - positive
    pos_weight = torch.clamp(negative / torch.clamp(positive, min=1.0), 1.0, 50.0)
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
    pred_prob = torch.sigmoid(pred)
    pred_flat = pred_prob.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + 1.0) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1.0
    )
    return bce + (1.0 - dice.mean())


@dataclass
class TrainConfig:
    dataset_dir: str
    output_dir: str
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    val_split: float
    seed: int
    num_workers: int
    device: str
    resume: str
    export_onnx: bool


def find_images(dataset_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"epoch {epoch:03d} train", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        preds = model(images)
        loss = dice_bce_loss(preds, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(model: UNet, loader: DataLoader, device: torch.device, epoch: int) -> float:
    model.eval()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"epoch {epoch:03d} val", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(images)
        loss = dice_bce_loss(preds, masks)
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, len(loader))


def save_checkpoint(
    model: UNet,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    is_best: bool,
) -> None:
    state_dict_path = output_dir / f"dust_unet_epoch{epoch:03d}.pth"
    torch.save(model.state_dict(), state_dict_path)

    training_state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(training_state, output_dir / "last_training_state.pt")
    shutil.copy2(state_dict_path, output_dir / "latest.pth")
    if is_best:
        shutil.copy2(state_dict_path, output_dir / "best.pth")


def export_onnx(model: UNet, output_dir: Path, image_size: int, device: torch.device) -> None:
    onnx_path = output_dir / "dust_unet.onnx"
    model.eval()
    export_model = UNetWithSigmoid(model).to(device).eval()
    example = torch.randn(1, 1, image_size, image_size, device=device)
    torch.onnx.export(
        export_model,
        example,
        onnx_path,
        input_names=["input"],
        output_names=["mask"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "mask": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"Exported ONNX model: {onnx_path}")

    package_dir = output_dir / "dust-negative-restore"
    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, package_dir / "dust_unet.onnx")
    (package_dir / "config.json").write_text(
        json.dumps(
            {
                "id": "dust-negative-restore",
                "name": "Dust and scratches negative restore",
                "description": "Detects dust and scratches in positive film scans.",
                "task": "negative-restore",
                "backend": "onnx",
                "arch": "unet",
                "num_inputs": 1,
                "attributes": {
                    "model_file": "dust_unet.onnx",
                    "tensor_layout": "NCHW",
                    "input": "luminance",
                    "output": "mask",
                    "height_dim": "height",
                    "width_dim": "width",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote darktable model package folder: {package_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Spotless Film U-Net dust detector."
    )
    parser.add_argument(
        "--dataset-dir",
        default="film-dataset",
        help="Directory containing clean film scans.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Directory for .pth checkpoints and metadata.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Square training size. Must be divisible by 16. Use 1024 for full notebook parity.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Optional last_training_state.pt checkpoint to resume from.",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export dust_unet.onnx after training for native/plugin integration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by 16 for the U-Net pooling path")
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError("--val-split must be in [0.0, 1.0)")

    seed_everything(args.seed)
    device = choose_device(args.device)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(dataset_dir)
    if not image_paths:
        raise FileNotFoundError(f"No training images found in {dataset_dir}")

    val_count = int(len(image_paths) * args.val_split)
    train_count = len(image_paths) - val_count
    if train_count <= 0:
        raise ValueError("Validation split leaves no training images")

    if val_count:
        generator = torch.Generator().manual_seed(args.seed)
        train_paths, val_paths = random_split(
            image_paths, [train_count, val_count], generator=generator
        )
        train_dataset = DustDataset(list(train_paths), args.image_size, augment=True)
        val_dataset = DustDataset(list(val_paths), args.image_size, augment=False)
    else:
        train_dataset = DustDataset(image_paths, args.image_size, augment=True)
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_dataset is not None
        else None
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("val_loss", best_val))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    config = TrainConfig(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        device=str(device),
        resume=args.resume,
        export_onnx=args.export_onnx,
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    print(f"Found {len(image_paths)} images: {train_count} train, {val_count} val")
    print(f"Training on {device}; writing checkpoints to {output_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = (
            validate(model, val_loader, device, epoch)
            if val_loader is not None
            else train_loss
        )
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        save_checkpoint(
            model, optimizer, output_dir, epoch, train_loss, val_loss, is_best=is_best
        )
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} best={best_val:.5f}"
        )

    if args.export_onnx:
        export_onnx(model, output_dir, args.image_size, device)

    print(f"Done. Use {output_dir / 'best.pth'} in src/weights/ or darktable tooling.")


if __name__ == "__main__":
    main()
