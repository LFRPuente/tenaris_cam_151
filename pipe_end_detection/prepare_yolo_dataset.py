from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path | None
    relative_stem: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a YOLOv11 dataset for pipe-end detection."
    )
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Treat images without labels as background images with empty YOLO labels.",
    )
    return parser.parse_args()


def ensure_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must add up to 1.0. Received {train_ratio} + {val_ratio} + {test_ratio} = {total}."
        )


def iter_images(images_root: Path) -> list[Path]:
    return sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_label_line(label_path: Path, line: str, line_number: int) -> None:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"{label_path}: line {line_number} must have 5 fields.")
    class_id = int(parts[0])
    if class_id != 0:
        raise ValueError(f"{label_path}: line {line_number} must use class 0 only.")
    coords = [float(value) for value in parts[1:]]
    for value in coords:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{label_path}: line {line_number} has normalized values outside [0, 1]."
            )
    _, _, width, height = coords
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{label_path}: line {line_number} must have positive width and height.")


def validate_label_file(label_path: Path) -> None:
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for index, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        validate_label_line(label_path, raw_line, index)


def collect_samples(images_root: Path, labels_root: Path, allow_missing_labels: bool) -> list[Sample]:
    samples: list[Sample] = []
    for image_path in iter_images(images_root):
        relative_path = image_path.relative_to(images_root)
        relative_stem = relative_path.with_suffix("")
        expected_label = (labels_root / relative_stem).with_suffix(".txt")
        if expected_label.exists():
            validate_label_file(expected_label)
            label_path: Path | None = expected_label
        elif allow_missing_labels:
            label_path = None
        else:
            raise FileNotFoundError(
                f"Missing label for image {image_path}. Expected {expected_label}."
            )
        samples.append(
            Sample(
                image_path=image_path,
                label_path=label_path,
                relative_stem=relative_stem,
            )
        )
    return samples


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    del test_ratio
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_yaml(output_root: Path) -> None:
    yaml_path = output_root.parent / "data.yaml"
    yaml_path.write_text(
        "path: .\n"
        "train: dataset/images/train\n"
        "val: dataset/images/val\n"
        "test: dataset/images/test\n\n"
        "names:\n"
        "  0: pipe_end\n",
        encoding="utf-8",
    )


def copy_split(split_name: str, samples: list[Sample], output_root: Path) -> None:
    images_dir = output_root / "images" / split_name
    labels_dir = output_root / "labels" / split_name
    for sample in samples:
        target_image = (images_dir / sample.relative_stem).with_suffix(sample.image_path.suffix.lower())
        target_label = (labels_dir / sample.relative_stem).with_suffix(".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, target_image)
        if sample.label_path is None:
            target_label.write_text("", encoding="utf-8")
        else:
            shutil.copy2(sample.label_path, target_label)


def main() -> None:
    args = parse_args()
    ensure_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    images_root = args.images_root.resolve()
    labels_root = args.labels_root.resolve()
    output_root = args.output_root.resolve()

    if not images_root.exists():
        raise FileNotFoundError(f"Images root does not exist: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels root does not exist: {labels_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(images_root, labels_root, args.allow_missing_labels)
    if not samples:
        raise ValueError("No images were found to build the dataset.")

    splits = split_samples(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    for split_name, split_samples_list in splits.items():
        copy_split(split_name, split_samples_list, output_root)

    write_yaml(output_root)

    print("YOLO dataset created successfully.")
    for split_name, split_samples_list in splits.items():
        print(f"- {split_name}: {len(split_samples_list)} images")
    print(f"- output: {output_root}")
    print(f"- yaml: {output_root.parent / 'data.yaml'}")


if __name__ == "__main__":
    main()
