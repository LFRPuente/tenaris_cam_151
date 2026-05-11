from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
PIPE_END_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO predictions into predictions/current/labels for app review.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained best.pt.")
    parser.add_argument("--images-root", type=Path, default=PIPE_END_ROOT / "annotation_pool" / "images")
    parser.add_argument("--output-root", type=Path, default=PIPE_END_ROOT / "predictions" / "current")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--include-labeled", action="store_true")
    parser.add_argument("--labels-root", type=Path, default=PIPE_END_ROOT / "annotation_pool" / "labels")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def has_label(label_path: Path) -> bool:
    return label_path.exists() and any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise SystemExit("Python package 'ultralytics' was not found. Install ultralytics first.") from exc

    images_root = args.images_root.resolve()
    labels_root = args.labels_root.resolve()
    output_root = args.output_root.resolve()
    prediction_labels_root = output_root / "labels"
    prediction_labels_root.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    for image_path in images_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = image_path.relative_to(images_root)
        label_path = (labels_root / rel).with_suffix(".txt")
        if not args.include_labeled and has_label(label_path):
            continue
        image_paths.append(image_path)

    if not image_paths:
        raise SystemExit("No images to predict.")

    print("Images queued for prediction:", len(image_paths))
    print("Weights:", args.weights)
    print("Output labels:", prediction_labels_root)
    if args.dry_run:
        return

    model = YOLO(str(args.weights))
    written = 0
    for image_path in image_paths:
        rel = image_path.relative_to(images_root)
        target_label = (prediction_labels_root / rel).with_suffix(".txt")
        target_label.parent.mkdir(parents=True, exist_ok=True)
        results = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
        lines: list[str] = []
        if results:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is not None and boxes.xywhn is not None:
                xywhn = boxes.xywhn.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                for class_id, coords, score in zip(cls, xywhn, conf):
                    if int(class_id) != 0:
                        continue
                    x, y, w, h = [float(value) for value in coords]
                    lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {float(score):.6f}")
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    print("Prediction label files written:", written)
    print("Open the annotation app and use 'Load AI boxes' on images with predictions.")


if __name__ == "__main__":
    main()
