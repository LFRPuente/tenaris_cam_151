from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_END_ROOT = REPO_ROOT / "pipe_end_detection"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from apps.pipe_end_annotator.annotate_app import (  # noqa: E402
    AppPaths,
    collect_annotated_samples,
    prepare_active_dataset,
    task_config,
)


def _label_has_boxes(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())


def _read_best_metrics(results_csv: Path) -> dict[str, str] | None:
    if not results_csv.exists():
        return None
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    if not rows:
        return None
    key = "metrics/mAP50-95(B)"
    rows.sort(key=lambda row: float((row.get(key) or "0").strip() or 0), reverse=True)
    return rows[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cam152 pipe_end from the YOLO11m base weights.")
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--project",
        type=Path,
        default=PIPE_END_ROOT / "runs" / "pipe_end_cam152_yolo11m_scratch",
    )
    parser.add_argument("--name", default="latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AppPaths(
        root=PIPE_END_ROOT,
        images_root=PIPE_END_ROOT / "annotation_pool" / "images",
        labels_root=PIPE_END_ROOT / "annotation_pool" / "labels",
        predictions_root=PIPE_END_ROOT / "predictions" / "current" / "labels",
        status_path=PIPE_END_ROOT / "annotation_pool" / "image_status.json",
        auto_train=False,
        min_train_images=4,
        train_epochs=args.epochs,
        train_imgsz=args.imgsz,
        train_batch=args.batch,
        base_model=args.model,
        train_device=args.device,
        roi_toml_151=None,
        roi_toml_152=None,
        raw_image_151=None,
        raw_image_152=None,
    )
    task = task_config(paths, "pipe_end_cam152")
    task.labels_root.mkdir(parents=True, exist_ok=True)
    task.predictions_root.mkdir(parents=True, exist_ok=True)
    task.work_root.mkdir(parents=True, exist_ok=True)
    task.project_root.mkdir(parents=True, exist_ok=True)
    args.project.mkdir(parents=True, exist_ok=True)

    samples = collect_annotated_samples(paths, task)
    positive_count = sum(1 for _, label_path, _ in samples if _label_has_boxes(label_path))
    negative_count = len(samples) - positive_count
    print(
        json.dumps(
            {
                "task": task.key,
                "samples": len(samples),
                "positive_images": positive_count,
                "negative_images": negative_count,
                "base_model": args.model,
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "batch": args.batch,
                "device": args.device,
            },
            indent=2,
        )
    )
    if len(samples) < paths.min_train_images:
        raise SystemExit(f"Only {len(samples)} cam152 training samples found.")

    data_yaml = prepare_active_dataset(paths, task, samples)
    stable_weights = task.model_path
    if stable_weights.exists():
        backup = stable_weights.with_name(
            f"best_before_cam152_yolo11m_scratch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        shutil.copy2(stable_weights, backup)
        print(f"BACKUP {stable_weights} -> {backup}")

    from ultralytics import YOLO  # type: ignore

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch),
        project=str(args.project),
        name=str(args.name),
        exist_ok=True,
        device=str(args.device),
        mosaic=0,
        fliplr=0,
        flipud=0,
        translate=0,
        scale=0,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        close_mosaic=0,
        workers=0,
    )

    run_dir = args.project / args.name
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(best_weights)
    stable_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, stable_weights)
    print(f"FINAL_COPIED {best_weights} -> {stable_weights}")

    best_metrics = _read_best_metrics(run_dir / "results.csv")
    if best_metrics:
        print("BEST_METRICS " + json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
