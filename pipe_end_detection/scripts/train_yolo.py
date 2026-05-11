from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPE_END_ROOT = SCRIPT_DIR.parent


def count_nonempty_labels(labels_root: Path) -> int:
    count = 0
    for path in labels_root.rglob("*.txt"):
        if any(line.strip() for line in path.read_text(encoding="utf-8").splitlines()):
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset and train a YOLO pipe_end model.")
    parser.add_argument("--images-root", type=Path, default=PIPE_END_ROOT / "annotation_pool" / "images")
    parser.add_argument("--labels-root", type=Path, default=PIPE_END_ROOT / "annotation_pool" / "labels")
    parser.add_argument("--output-root", type=Path, default=PIPE_END_ROOT / "dataset")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default=str(PIPE_END_ROOT / "runs" / "pipe_end"))
    parser.add_argument("--name", default="train")
    parser.add_argument(
        "--include-unlabeled-background",
        action="store_true",
        help="Include images without boxes as background examples. Default trains only on labels with boxes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_root = args.labels_root.resolve()
    annotated = count_nonempty_labels(labels_root)
    if annotated < 5:
        raise SystemExit(
            f"Only {annotated} annotated images found. Annotate at least 15-25 images before first training."
        )

    prepare_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_yolo_dataset.py"),
        "--images-root",
        str(args.images_root),
        "--labels-root",
        str(args.labels_root),
        "--output-root",
        str(args.output_root),
    ]
    if args.include_unlabeled_background:
        prepare_cmd.append("--allow-missing-labels")
    else:
        prepare_cmd.append("--require-box-labels")
    train_cmd = [
        "yolo",
        "detect",
        "train",
        f"data={args.output_root.resolve().parent / 'data.yaml'}",
        f"model={args.model}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"project={args.project}",
        f"name={args.name}",
    ]

    print("Annotated non-empty labels:", annotated)
    print("Prepare command:", " ".join(prepare_cmd))
    print("Train command:", " ".join(train_cmd))
    if args.dry_run:
        return
    subprocess.check_call(prepare_cmd)
    if shutil.which("yolo") is None:
        raise SystemExit("Ultralytics CLI 'yolo' was not found. Install ultralytics or run the printed train command manually.")
    subprocess.check_call(train_cmd)


if __name__ == "__main__":
    main()
