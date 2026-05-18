from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID = 0
CLASS_NAME = "tube_bundle"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    rel_stem: Path
    box_xyxy: tuple[float, float, float, float] | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_box(label_path: Path, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not label_path.exists():
        return None
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id != CLASS_ID:
            continue
        xc, yc, bw, bh = [float(v) for v in parts[1:5]]
        x1 = max(0.0, (xc - bw / 2.0) * width)
        y1 = max(0.0, (yc - bh / 2.0) * height)
        x2 = min(float(width - 1), (xc + bw / 2.0) * width)
        y2 = min(float(height - 1), (yc + bh / 2.0) * height)
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
    return None


def image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    return int(w), int(h)


def load_bad_warps(status_path: Path) -> set[str]:
    if not status_path.exists():
        return set()
    data = json.loads(status_path.read_text(encoding="utf-8"))
    bad = data.get("bad_warp", {}) if isinstance(data, dict) else {}
    return {str(key) for key, value in bad.items() if value}


def is_bad_warp(rel_with_suffix: Path, bad_warps: set[str]) -> bool:
    if rel_with_suffix.as_posix() in bad_warps:
        return True
    rel_stem = rel_with_suffix.with_suffix("")
    return any(rel_stem.with_suffix(suffix).as_posix() in bad_warps for suffix in IMAGE_SUFFIXES)


def collect_samples(images_root: Path, labels_root: Path, bad_warps: set[str]) -> tuple[list[Sample], list[Sample]]:
    positives: list[Sample] = []
    negatives: list[Sample] = []
    for image_path in sorted(images_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel_with_suffix = image_path.relative_to(images_root)
        if is_bad_warp(rel_with_suffix, bad_warps):
            continue
        rel_stem = rel_with_suffix.with_suffix("")
        label_path = (labels_root / rel_stem).with_suffix(".txt")
        width, height = image_size(image_path)
        box = read_box(label_path, width, height)
        sample = Sample(image_path=image_path, label_path=label_path, rel_stem=rel_stem, box_xyxy=box)
        if box is None:
            negatives.append(sample)
        else:
            positives.append(sample)
    return positives, negatives


def choose_limited_negatives(
    negatives: list[Sample],
    *,
    train_count: int,
    val_count: int,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    by_camera: dict[str, list[Sample]] = {}
    for sample in negatives:
        camera = sample.rel_stem.parts[0] if sample.rel_stem.parts else ""
        by_camera.setdefault(camera, []).append(sample)
    for group in by_camera.values():
        rng.shuffle(group)

    val_neg: list[Sample] = []
    train_neg: list[Sample] = []
    cameras = sorted(by_camera)

    while len(val_neg) < val_count and any(by_camera.values()):
        for camera in cameras:
            if len(val_neg) >= val_count:
                break
            group = by_camera.get(camera, [])
            if group:
                val_neg.append(group.pop())

    while len(train_neg) < train_count and any(by_camera.values()):
        for camera in cameras:
            if len(train_neg) >= train_count:
                break
            group = by_camera.get(camera, [])
            if group:
                train_neg.append(group.pop())

    return train_neg, val_neg


def polygon_from_mask(mask: np.ndarray, width: int, height: int, epsilon_ratio: float) -> list[tuple[float, float]]:
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, epsilon_ratio * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        return []
    points: list[tuple[float, float]] = []
    for x, y in approx:
        xn = min(1.0, max(0.0, float(x) / float(width)))
        yn = min(1.0, max(0.0, float(y) / float(height)))
        points.append((xn, yn))
    return points


def rectangle_polygon(box: tuple[float, float, float, float], width: int, height: int) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = box
    return [
        (x1 / width, y1 / height),
        (x2 / width, y1 / height),
        (x2 / width, y2 / height),
        (x1 / width, y2 / height),
    ]


def write_seg_label(path: Path, points: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        path.write_text("", encoding="utf-8")
        return
    coords = " ".join(f"{value:.6f}" for point in points for value in point)
    path.write_text(f"{CLASS_ID} {coords}\n", encoding="utf-8")


def draw_overlay(
    *,
    image_path: Path,
    mask: np.ndarray | None,
    box: tuple[float, float, float, float] | None,
    polygon: list[tuple[float, float]],
    output_path: Path,
) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    overlay = img.copy()
    if mask is not None:
        mask_resized = mask
        if mask_resized.shape[:2] != (h, w):
            mask_resized = cv2.resize(mask_resized.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        color = np.zeros_like(img)
        color[:, :, 1] = 220
        overlay = np.where(mask_resized[..., None].astype(bool), (0.55 * overlay + 0.45 * color).astype(np.uint8), overlay)
    if polygon:
        pts = np.array([[[int(x * w), int(y * h)] for x, y in polygon]], dtype=np.int32)
        cv2.polylines(overlay, pts, True, (0, 255, 255), 3)
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def generate_positive_label(
    *,
    model: SAM,
    sample: Sample,
    target_label: Path,
    overlay_path: Path | None,
    device: str,
    epsilon_ratio: float,
) -> dict:
    width, height = image_size(sample.image_path)
    if sample.box_xyxy is None:
        write_seg_label(target_label, [])
        return {"status": "negative"}

    results = model.predict(str(sample.image_path), bboxes=[list(sample.box_xyxy)], device=device, verbose=False)
    mask: np.ndarray | None = None
    if results and getattr(results[0], "masks", None) is not None:
        data = results[0].masks.data
        if data is not None and len(data) > 0:
            mask = data[0].cpu().numpy().astype(np.uint8)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    points = polygon_from_mask(mask, width, height, epsilon_ratio) if mask is not None else []
    fallback = False
    if len(points) < 3:
        points = rectangle_polygon(sample.box_xyxy, width, height)
        fallback = True

    write_seg_label(target_label, points)
    if overlay_path is not None:
        draw_overlay(image_path=sample.image_path, mask=mask, box=sample.box_xyxy, polygon=points, output_path=overlay_path)

    return {
        "status": "ok",
        "fallback_rectangle": fallback,
        "point_count": len(points),
        "mask_area_px": int(mask.sum()) if mask is not None else 0,
    }


def copy_image(sample: Sample, split: str, dataset_root: Path) -> Path:
    target = (dataset_root / "images" / split / sample.rel_stem).with_suffix(sample.image_path.suffix.lower())
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image_path, target)
    return target


def build_dataset(args: argparse.Namespace) -> dict:
    root = args.repo_root.resolve()
    images_root = (root / args.images_root).resolve()
    labels_root = (root / args.box_labels_root).resolve()
    output_root = (root / args.output_root).resolve()
    dataset_root = output_root / "dataset"
    overlays_root = output_root / "overlays"
    bad_warps = load_bad_warps((root / args.status_path).resolve())

    positives, negatives = collect_samples(images_root, labels_root, bad_warps)
    rng = random.Random(args.seed)
    rng.shuffle(positives)
    val_pos_count = max(1, round(args.val_ratio * len(positives))) if len(positives) >= 8 else len(positives)
    val_pos = positives[:val_pos_count]
    train_pos = positives[val_pos_count:]
    train_neg, val_neg = choose_limited_negatives(
        negatives,
        train_count=args.train_negatives,
        val_count=args.val_negatives,
        seed=args.seed,
    )

    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    if overlays_root.exists():
        shutil.rmtree(overlays_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model = SAM(args.sam_model)
    summary: dict[str, object] = {
        "sam_model": args.sam_model,
        "device": args.device,
        "all_positive_images": len(positives),
        "all_candidate_negatives": len(negatives),
        "train_positive_images": len(train_pos),
        "train_negative_images": len(train_neg),
        "val_positive_images": len(val_pos),
        "val_negative_images": len(val_neg),
        "samples": [],
    }
    sample_rows: list[dict] = []
    overlay_budget = int(args.overlay_count)

    for split, split_samples in (
        ("train", train_pos + train_neg),
        ("val", val_pos + val_neg),
    ):
        rng.shuffle(split_samples)
        for sample in split_samples:
            copy_image(sample, split, dataset_root)
            target_label = (dataset_root / "labels" / split / sample.rel_stem).with_suffix(".txt")
            if sample.box_xyxy is None:
                write_seg_label(target_label, [])
                sample_rows.append({"split": split, "rel": sample.rel_stem.as_posix(), "status": "negative"})
                continue

            overlay_path = None
            if overlay_budget > 0:
                overlay_path = (overlays_root / split / sample.rel_stem).with_suffix(".jpg")
                overlay_budget -= 1
            result = generate_positive_label(
                model=model,
                sample=sample,
                target_label=target_label,
                overlay_path=overlay_path,
                device=args.device,
                epsilon_ratio=args.epsilon_ratio,
            )
            sample_rows.append({"split": split, "rel": sample.rel_stem.as_posix(), **result})

    data_yaml = output_root / "data_active.yaml"
    data_yaml.write_text(
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"  {CLASS_ID}: {CLASS_NAME}\n",
        encoding="utf-8",
    )

    summary["train_images"] = len(train_pos) + len(train_neg)
    summary["val_images"] = len(val_pos) + len(val_neg)
    summary["data_yaml"] = str(data_yaml)
    summary["dataset_root"] = str(dataset_root)
    summary["samples"] = sample_rows
    summary_path = output_root / "sam2_seg_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO segmentation labels for tube bundles using SAM2.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--images-root", type=Path, default=Path("pipe_end_detection/annotation_pool/images"))
    parser.add_argument("--box-labels-root", type=Path, default=Path("tube_bundle_detection/annotation_pool/labels"))
    parser.add_argument("--status-path", type=Path, default=Path("pipe_end_detection/annotation_pool/image_status.json"))
    parser.add_argument("--output-root", type=Path, default=Path("tube_bundle_seg_detection/active_training_sam2_s"))
    parser.add_argument("--sam-model", default="sam2_s.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--train-negatives", type=int, default=5)
    parser.add_argument("--val-negatives", type=int, default=3)
    parser.add_argument("--epsilon-ratio", type=float, default=0.003)
    parser.add_argument("--overlay-count", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    summary = build_dataset(parse_args())
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
