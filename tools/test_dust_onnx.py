#!/usr/bin/env python3
"""Run the dust/scratch ONNX model on an image and write debug previews."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_rgb(path: Path, image: np.ndarray) -> None:
    image = np.clip(image, 0.0, 1.0)
    Image.fromarray((image * 255.0 + 0.5).astype(np.uint8), "RGB").save(path)


def _save_gray(path: Path, image: np.ndarray) -> None:
    image = np.clip(image, 0.0, 1.0)
    Image.fromarray((image * 255.0 + 0.5).astype(np.uint8), "L").save(path)


def _pad_to_multiple(image: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape
    padded_height = ((height + multiple - 1) // multiple) * multiple
    padded_width = ((width + multiple - 1) // multiple) * multiple
    pad_y = padded_height - height
    pad_x = padded_width - width
    if pad_y == 0 and pad_x == 0:
        return image, (height, width)
    return np.pad(image, ((0, pad_y), (0, pad_x)), mode="reflect"), (height, width)


def _simple_repair(rgb: np.ndarray, mask: np.ndarray, threshold: float, radius: int) -> np.ndarray:
    height, width, _ = rgb.shape
    out = rgb.copy()
    detected = mask >= threshold

    for y in range(height):
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        for x in range(width):
            if not detected[y, x]:
                continue

            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            patch = rgb[y0:y1, x0:x1]
            patch_detected = detected[y0:y1, x0:x1]
            good = ~patch_detected

            if np.any(good):
                out[y, x] = patch[good].mean(axis=0)

    return out


def _run_model(session, input_name: str, output_name: str, luminance: np.ndarray, multiple: int) -> np.ndarray:
    padded_luminance, original_shape = _pad_to_multiple(luminance, multiple)
    tensor = padded_luminance[None, None, :, :].astype(np.float32)
    padded_mask = session.run([output_name], {input_name: tensor})[0][0, 0].astype(np.float32)
    height, width = original_shape
    return padded_mask[:height, :width]


def _run_model_tiled(
    session,
    input_name: str,
    output_name: str,
    luminance: np.ndarray,
    multiple: int,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    height, width = luminance.shape
    mask_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    tile_size = max(multiple, tile_size)
    overlap = max(0, min(overlap, tile_size // 2 - 1))
    step = tile_size - 2 * overlap
    if step <= 0:
        step = tile_size
        overlap = 0

    y_positions = list(range(0, height, step))
    x_positions = list(range(0, width, step))
    if y_positions[-1] + tile_size < height:
        y_positions.append(height - tile_size)
    if x_positions[-1] + tile_size < width:
        x_positions.append(width - tile_size)
    y_positions = sorted(set(max(0, y) for y in y_positions))
    x_positions = sorted(set(max(0, x) for x in x_positions))

    total = len(y_positions) * len(x_positions)
    done = 0
    for y in y_positions:
        for x in x_positions:
            y0 = y
            x0 = x
            y1 = min(height, y0 + tile_size)
            x1 = min(width, x0 + tile_size)
            tile = luminance[y0:y1, x0:x1]
            tile_mask = _run_model(session, input_name, output_name, tile, multiple)

            weight = np.ones(tile_mask.shape, dtype=np.float32)
            if overlap > 0:
                yy = np.minimum(np.arange(tile_mask.shape[0]), np.arange(tile_mask.shape[0])[::-1])
                xx = np.minimum(np.arange(tile_mask.shape[1]), np.arange(tile_mask.shape[1])[::-1])
                wy = np.clip((yy + 1) / overlap, 0.0, 1.0)
                wx = np.clip((xx + 1) / overlap, 0.0, 1.0)
                weight = np.minimum(wy[:, None], wx[None, :]).astype(np.float32)

            mask_sum[y0:y1, x0:x1] += tile_mask * weight
            weight_sum[y0:y1, x0:x1] += weight
            done += 1
            print(f"tile {done}/{total}", end="\r", flush=True)

    print()
    return mask_sum / np.maximum(weight_sum, 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="input JPEG/PNG/TIFF image")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("checkpoints/dust_unet.onnx"),
        help="ONNX model path",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("dust_onnx_debug"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--multiple", type=int, default=16, help="pad inference input to this multiple")
    parser.add_argument("--tile-size", type=int, default=512, help="run inference in tiles of this size")
    parser.add_argument("--overlap", type=int, default=64, help="tile overlap in pixels")
    parser.add_argument("--full-image", action="store_true", help="disable tiled inference")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()

    import onnxruntime as ort

    rgb = _load_image(args.image)
    luminance = (
        0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)
    multiple = max(1, args.multiple)

    session = ort.InferenceSession(str(args.model), providers=[args.provider])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    if args.full_image:
        mask = _run_model(session, input_name, output_name, luminance, multiple)
        padded_luminance, _ = _pad_to_multiple(luminance, multiple)
        inferred_shape = (1, 1, padded_luminance.shape[0], padded_luminance.shape[1])
    else:
        mask = _run_model_tiled(
            session,
            input_name,
            output_name,
            luminance,
            multiple,
            args.tile_size,
            args.overlap,
        )
        inferred_shape = (1, 1, args.tile_size, args.tile_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    overlay = rgb.copy()
    detected = mask >= args.threshold
    overlay[detected] = 0.65 * overlay[detected] + np.array([0.35, 0.0, 0.0])

    repaired = _simple_repair(rgb, mask, args.threshold, max(1, args.radius))

    _save_gray(args.out_dir / f"{stem}_mask.png", mask)
    _save_rgb(args.out_dir / f"{stem}_red_overlay.png", overlay)
    _save_rgb(args.out_dir / f"{stem}_repaired_preview.png", repaired)

    print(f"input: {input_name} {inferred_shape} float32")
    print(f"original image: {rgb.shape[0]}x{rgb.shape[1]}")
    print(f"output: {output_name} {mask.shape} float32")
    print(
        "mask stats: "
        f"min={mask.min():.6f} max={mask.max():.6f} "
        f"mean={mask.mean():.6f} pixels>={args.threshold:g}={int(detected.sum())}"
    )
    print(
        "percentiles: "
        f"p90={np.percentile(mask, 90):.6f} "
        f"p95={np.percentile(mask, 95):.6f} "
        f"p99={np.percentile(mask, 99):.6f} "
        f"p99.9={np.percentile(mask, 99.9):.6f}"
    )
    for t in (0.05, 0.1, 0.2, 0.3, 0.4):
        print(f"pixels>={t:g}: {int((mask >= t).sum())}")
    print(f"wrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
