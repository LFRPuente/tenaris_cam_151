from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID = 0
CLASS_NAME = "pipe_end"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PIPE_END_ROOT = REPO_ROOT / "pipe_end_detection"


@dataclass(frozen=True)
class AppPaths:
    root: Path
    images_root: Path
    labels_root: Path
    predictions_root: Path
    status_path: Path
    auto_train: bool
    min_train_images: int
    train_epochs: int
    train_imgsz: int
    train_batch: int
    base_model: str
    train_device: str
    roi_toml_151: Path | None
    roi_toml_152: Path | None
    raw_image_151: Path | None
    raw_image_152: Path | None


@dataclass(frozen=True)
class AnnotationTask:
    key: str
    label: str
    class_name: str
    labels_root: Path
    predictions_root: Path
    model_path: Path
    work_root: Path
    project_root: Path
    camera_filter: str | None = None
    max_boxes: int | None = None
    use_pipe_end_inference: bool = False
    single_image_conf: float = 0.20
    include_unlabeled_negatives: bool = False
    allow_negative_labels: bool = False

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "class_id": CLASS_ID,
            "class_name": self.class_name,
            "camera_filter": self.camera_filter,
            "max_boxes": self.max_boxes,
            "model_path": repo_display_path(self.model_path),
            "labels_root": repo_display_path(self.labels_root),
            "predictions_root": repo_display_path(self.predictions_root),
            "include_unlabeled_negatives": self.include_unlabeled_negatives,
            "allow_negative_labels": self.allow_negative_labels,
        }


class TrainingState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.pending = False
        self.status = "idle"
        self.message = "No training has run in this session."
        self.annotated_images = 0
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.weights_path: str | None = None
        self.error: str | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "pending": self.pending,
                "status": self.status,
                "message": self.message,
                "annotated_images": self.annotated_images,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "weights_path": self.weights_path,
                "error": self.error,
            }


TRAINING_STATE = TrainingState()
SAM_BOUNDARY_LOCK = threading.Lock()
SAM_BOUNDARY_MODELS: dict[str, object] = {}
SPACING_STATS_CACHE: dict[tuple[str, str, int, int], dict | None] = {}


def active_model_path() -> Path:
    return REPO_ROOT / "models" / "pipe_end_active" / "best.pt"


def cam152_pipe_end_model_path() -> Path:
    return REPO_ROOT / "models" / "pipe_end_cam152_active" / "best.pt"


def repo_display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def task_config(paths: AppPaths, task_key: str | None = None) -> AnnotationTask:
    key = (task_key or "pipe_end").strip().lower()
    if key == "pipe_end":
        return AnnotationTask(
            key="pipe_end",
            label="Pipe ends: cam151 model",
            class_name=CLASS_NAME,
            labels_root=paths.labels_root,
            predictions_root=paths.predictions_root,
            model_path=active_model_path(),
            work_root=paths.root / "active_training",
            project_root=paths.root / "runs" / "pipe_end_active",
            camera_filter="cam151",
            use_pipe_end_inference=True,
            single_image_conf=_single_image_predict_conf(),
            allow_negative_labels=True,
        )
    if key == "pipe_end_cam152":
        return AnnotationTask(
            key="pipe_end_cam152",
            label="Pipe ends: cam152 model",
            class_name=CLASS_NAME,
            labels_root=paths.labels_root,
            predictions_root=paths.root / "predictions" / "cam152_pipe_end" / "labels",
            model_path=cam152_pipe_end_model_path(),
            work_root=paths.root / "active_training_cam152",
            project_root=paths.root / "runs" / "pipe_end_cam152_active",
            camera_filter="cam152",
            use_pipe_end_inference=True,
            single_image_conf=_single_image_predict_conf(),
            allow_negative_labels=True,
        )
    raise ValueError(f"Unknown annotation task: {task_key!r}")


def list_tasks(paths: AppPaths) -> list[AnnotationTask]:
    return [task_config(paths, key) for key in ("pipe_end", "pipe_end_cam152")]


def normalize_rel(path: str) -> Path:
    decoded = unquote(path).replace("\\", "/")
    normalized = posixpath.normpath(decoded).lstrip("/")
    if normalized == "." or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"Unsafe relative path: {path}")
    return Path(*normalized.split("/"))


def rel_to_url(path: Path) -> str:
    return quote(path.as_posix())


def load_status(status_path: Path) -> dict:
    if not status_path.exists():
        return {"bad_warp": {}, "negative_labels": {}, "notes": {}}
    data = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"bad_warp": {}, "negative_labels": {}, "notes": {}}
    data.setdefault("bad_warp", {})
    data.setdefault("negative_labels", {})
    data.setdefault("notes", {})
    return data


def save_status(status_path: Path, data: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def status_rel_keys(rel: Path) -> list[str]:
    if rel.suffix.lower() in IMAGE_SUFFIXES:
        return [rel.as_posix()]
    return [rel.with_suffix(suffix).as_posix() for suffix in sorted(IMAGE_SUFFIXES)]


def is_negative_annotation(status: dict, task: AnnotationTask, rel: Path) -> bool:
    task_negatives = status.get("negative_labels", {}).get(task.key, {})
    if not isinstance(task_negatives, dict):
        return False
    return any(bool(task_negatives.get(key, False)) for key in status_rel_keys(rel))


def set_negative_annotation(status: dict, task: AnnotationTask, rel: Path, flag: bool) -> bool:
    negative_labels = status.setdefault("negative_labels", {})
    if not isinstance(negative_labels, dict):
        negative_labels = {}
        status["negative_labels"] = negative_labels
    task_negatives = negative_labels.setdefault(task.key, {})
    if not isinstance(task_negatives, dict):
        task_negatives = {}
        negative_labels[task.key] = task_negatives
    keys = status_rel_keys(rel)
    if flag:
        task_negatives[keys[0]] = True
    else:
        for key in keys:
            task_negatives.pop(key, None)
    return is_negative_annotation(status, task, rel)


def yolo_to_boxes(label_path: Path, image_width: int, image_height: int) -> list[dict]:
    if not label_path.exists():
        return []
    boxes: list[dict] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in {5, 6}:
            continue
        class_id, x_center, y_center, width, height = parts[:5]
        if int(float(class_id)) != CLASS_ID:
            continue
        xc = float(x_center) * image_width
        yc = float(y_center) * image_height
        w = float(width) * image_width
        h = float(height) * image_height
        boxes.append(
            {
                "class_id": CLASS_ID,
                "x": max(0.0, xc - 0.5 * w),
                "y": max(0.0, yc - 0.5 * h),
                "w": max(0.0, w),
                "h": max(0.0, h),
                "conf": float(parts[5]) if len(parts) == 6 else None,
            }
        )
    return boxes


def boxes_to_yolo(boxes: list[dict], image_width: int, image_height: int) -> str:
    lines: list[str] = []
    for box in boxes:
        x = max(0.0, min(float(image_width), float(box.get("x", 0.0))))
        y = max(0.0, min(float(image_height), float(box.get("y", 0.0))))
        w = max(0.0, min(float(image_width) - x, float(box.get("w", 0.0))))
        h = max(0.0, min(float(image_height) - y, float(box.get("h", 0.0))))
        if w <= 0.0 or h <= 0.0:
            continue
        xc = (x + 0.5 * w) / float(image_width)
        yc = (y + 0.5 * h) / float(image_height)
        wn = w / float(image_width)
        hn = h / float(image_height)
        lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _box_to_xyxy(box: dict, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(image_width), float(box.get("x", 0.0))))
    y1 = max(0.0, min(float(image_height), float(box.get("y", 0.0))))
    x2 = max(0.0, min(float(image_width), x1 + float(box.get("w", 0.0))))
    y2 = max(0.0, min(float(image_height), y1 + float(box.get("h", 0.0))))
    return x1, y1, x2, y2


def _largest_yolo_box_xyxy(label_path: Path, image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    boxes = yolo_to_boxes(label_path, image_width, image_height)
    if not boxes:
        return None
    best = max(boxes, key=lambda box: float(box.get("w", 0.0)) * float(box.get("h", 0.0)))
    return _box_to_xyxy(best, image_width, image_height)


def _median_float(values: list[float]) -> float | None:
    clean = sorted(float(value) for value in values if float(value) > 1e-6)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float(0.5 * (clean[mid - 1] + clean[mid]))


def _mean_variance(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(float(value) for value in values) / float(len(values))
    variance = sum((float(value) - mean) ** 2 for value in values) / float(len(values))
    return float(mean), float(variance)


def _image_path_for_label_stem(paths: AppPaths, rel_stem: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = (paths.images_root / rel_stem).with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _task_pipe_end_spacing_stats(paths: AppPaths, task: AnnotationTask) -> dict | None:
    if not task.use_pipe_end_inference:
        return None
    label_files = [
        path
        for path in task.labels_root.rglob("*.txt")
        if path.is_file()
        and (
            not task.camera_filter
            or (
                (rel := path.relative_to(task.labels_root).with_suffix("")).parts
                and rel.parts[0] == task.camera_filter
            )
        )
    ]
    if not label_files:
        return None
    latest_mtime = max(path.stat().st_mtime_ns for path in label_files)
    cache_key = (task.key, task.camera_filter or "", len(label_files), int(latest_mtime))
    if cache_key in SPACING_STATS_CACHE:
        return SPACING_STATS_CACHE[cache_key]

    spacings: list[float] = []
    sample_count = 0
    for label_path in label_files:
        rel_stem = label_path.relative_to(task.labels_root).with_suffix("")
        image_path = _image_path_for_label_stem(paths, rel_stem)
        if image_path is None:
            continue
        try:
            width, height = image_size(image_path)
            boxes = yolo_to_boxes(label_path, width, height)
        except Exception:
            continue
        if len(boxes) < 2:
            continue
        ys = sorted(float(box.get("y", 0.0)) + 0.5 * float(box.get("h", 0.0)) for box in boxes)
        gaps = [ys[idx + 1] - ys[idx] for idx in range(len(ys) - 1) if ys[idx + 1] > ys[idx]]
        if gaps:
            sample_count += 1
            spacings.extend(gaps)

    raw_median = _median_float(spacings)
    if raw_median is None or len(spacings) < 10:
        SPACING_STATS_CACHE[cache_key] = None
        return None

    central = [gap for gap in spacings if 0.35 * raw_median <= gap <= 2.50 * raw_median]
    median_gap = _median_float(central) or raw_median
    mean_gap, variance_gap = _mean_variance(central)
    deviations = [abs(gap - median_gap) for gap in central]
    mad = _median_float(deviations) or 0.0
    robust_sigma = 1.4826 * float(mad)
    std_gap = float(variance_gap) ** 0.5
    allowed_gap = max(1.65 * float(median_gap), float(median_gap) + 4.0 * robust_sigma)
    if std_gap > 0:
        allowed_gap = max(allowed_gap, float(mean_gap) + 2.0 * std_gap)
    allowed_gap = min(allowed_gap, 3.0 * float(median_gap))

    stats = {
        "source": "annotation_spacing",
        "sample_image_count": int(sample_count),
        "spacing_count": int(len(central)),
        "raw_spacing_count": int(len(spacings)),
        "median_gap_px": float(median_gap),
        "mean_gap_px": float(mean_gap),
        "variance_gap_px": float(variance_gap),
        "std_gap_px": float(std_gap),
        "mad_gap_px": float(mad),
        "allowed_gap_px": float(allowed_gap),
    }
    SPACING_STATS_CACHE[cache_key] = stats
    return stats


def image_size(path: Path) -> tuple[int, int]:
    # Pillow is not required. Use OpenCV if installed; otherwise parse enough JPEG/PNG metadata.
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass

    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if suffix in {".jpg", ".jpeg"}:
        i = 2
        while i < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                length = int.from_bytes(data[i : i + 2], "big")
                segment = data[i : i + length]
                h = int.from_bytes(segment[3:5], "big")
                w = int.from_bytes(segment[5:7], "big")
                return w, h
            if marker in {0xD8, 0xD9}:
                continue
            length = int.from_bytes(data[i : i + 2], "big")
            i += length
    raise ValueError(f"Could not read image dimensions: {path}")


def build_image_items(paths: AppPaths, task: AnnotationTask | None = None) -> list[dict]:
    task = task or task_config(paths)
    items: list[dict] = []
    status = load_status(paths.status_path)
    bad_warp = status.get("bad_warp", {})
    for image_path in sorted(paths.images_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = image_path.relative_to(paths.images_root)
        if task.camera_filter and (not rel.parts or rel.parts[0] != task.camera_filter):
            continue
        label_path = (task.labels_root / rel).with_suffix(".txt")
        pred_path = (task.predictions_root / rel).with_suffix(".txt")
        rel_key = rel.as_posix()
        width, height = image_size(image_path)
        labeled_count = 0
        if label_path.exists():
            labeled_count = len([line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        negative = is_negative_annotation(status, task, rel)
        prediction_count = 0
        if pred_path.exists():
            prediction_count = len([line for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        items.append(
            {
                "rel": rel_key,
                "name": image_path.name,
                "camera": rel.parts[0] if rel.parts else "",
                "url": f"/image?path={rel_to_url(rel)}",
                "task": task.key,
                "label_rel": repo_display_path(label_path),
                "prediction_rel": repo_display_path(pred_path),
                "width": width,
                "height": height,
                "box_count": labeled_count,
                "prediction_count": prediction_count,
                "labeled": labeled_count > 0 or negative,
                "negative": negative,
                "has_predictions": prediction_count > 0,
                "bad_warp": bool(bad_warp.get(rel_key, False)),
            }
        )
    return items


def label_has_boxes(label_path: Path) -> bool:
    return label_path.exists() and any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())


def collect_annotated_samples(paths: AppPaths, task: AnnotationTask) -> list[tuple[Path, Path, Path]]:
    status = load_status(paths.status_path)
    bad_warp = status.get("bad_warp", {})
    samples: list[tuple[Path, Path, Path]] = []
    if task.include_unlabeled_negatives:
        for image_path in sorted(paths.images_root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel = image_path.relative_to(paths.images_root).with_suffix("")
            if task.camera_filter and (not rel.parts or rel.parts[0] != task.camera_filter):
                continue
            if bad_warp.get(rel.with_suffix(".jpg").as_posix(), False) or bad_warp.get(
                rel.with_suffix(".jpeg").as_posix(), False
            ):
                continue
            label_path = (task.labels_root / rel).with_suffix(".txt")
            samples.append((image_path, label_path, rel))
        return samples

    for label_path in sorted(task.labels_root.rglob("*.txt")):
        rel = label_path.relative_to(task.labels_root).with_suffix("")
        if not label_has_boxes(label_path) and not is_negative_annotation(status, task, rel):
            continue
        if task.camera_filter and (not rel.parts or rel.parts[0] != task.camera_filter):
            continue
        if bad_warp.get(rel.with_suffix(".jpg").as_posix(), False) or bad_warp.get(rel.with_suffix(".jpeg").as_posix(), False):
            continue
        image_path = None
        for suffix in IMAGE_SUFFIXES:
            candidate = (paths.images_root / rel).with_suffix(suffix)
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        samples.append((image_path, label_path, rel))
    return samples


def copy_training_split(samples: list[tuple[Path, Path, Path]], split: str, dataset_root: Path) -> None:
    for image_path, label_path, rel in samples:
        target_image = (dataset_root / "images" / split / rel).with_suffix(image_path.suffix.lower())
        target_label = (dataset_root / "labels" / split / rel).with_suffix(".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        if label_path.exists():
            shutil.copy2(label_path, target_label)
        else:
            target_label.write_text("", encoding="utf-8")


def prepare_active_dataset(paths: AppPaths, task: AnnotationTask, samples: list[tuple[Path, Path, Path]]) -> Path:
    work_root = task.work_root
    dataset_root = work_root / "dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    shuffled = list(samples)
    random.Random(42).shuffle(shuffled)
    if len(shuffled) < 8:
        train_samples = shuffled
        val_samples = shuffled
    else:
        val_count = max(1, int(round(0.2 * len(shuffled))))
        val_samples = shuffled[:val_count]
        train_samples = shuffled[val_count:]
    copy_training_split(train_samples, "train", dataset_root)
    copy_training_split(val_samples, "val", dataset_root)

    data_yaml = work_root / "data_active.yaml"
    data_yaml.write_text(
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"  {CLASS_ID}: {task.class_name}\n",
        encoding="utf-8",
    )
    return data_yaml


def schedule_training(paths: AppPaths, reason: str, task_key: str | None = None) -> dict:
    task = task_config(paths, task_key)
    with TRAINING_STATE.lock:
        if TRAINING_STATE.running:
            return {"scheduled": False, "reason": "AI job already running", "status": TRAINING_STATE.status}
        TRAINING_STATE.running = True
        TRAINING_STATE.pending = False
        TRAINING_STATE.status = "queued"
        TRAINING_STATE.message = f"Queued {task.label} training after {reason}."
        TRAINING_STATE.error = None
    thread = threading.Thread(target=training_worker, args=(paths, task.key), daemon=True)
    thread.start()
    return {"scheduled": True, "pending": False, "task": task.to_json()}


def training_worker(paths: AppPaths, task_key: str) -> None:
    try:
        task = task_config(paths, task_key)
        task.labels_root.mkdir(parents=True, exist_ok=True)
        task.predictions_root.mkdir(parents=True, exist_ok=True)
        task.work_root.mkdir(parents=True, exist_ok=True)
        task.project_root.mkdir(parents=True, exist_ok=True)
        samples = collect_annotated_samples(paths, task)
        with TRAINING_STATE.lock:
            TRAINING_STATE.annotated_images = len(samples)
        if len(samples) < paths.min_train_images:
            with TRAINING_STATE.lock:
                TRAINING_STATE.status = "skipped"
                TRAINING_STATE.message = (
                    f"{task.label}: need at least {paths.min_train_images} annotated images; currently {len(samples)}."
                )
                TRAINING_STATE.finished_at = time.time()
            return

        with TRAINING_STATE.lock:
            TRAINING_STATE.status = "training"
            TRAINING_STATE.message = f"Training {task.label} on {len(samples)} annotated images..."
            TRAINING_STATE.started_at = time.time()
            TRAINING_STATE.finished_at = None
            TRAINING_STATE.error = None

        data_yaml = prepare_active_dataset(paths, task, samples)
        stable_weights = task.model_path
        model_arg = _initial_training_model(task, paths.base_model)

        from ultralytics import YOLO  # type: ignore

        model = YOLO(model_arg)
        model.train(
            data=str(data_yaml),
            imgsz=int(paths.train_imgsz),
            epochs=int(paths.train_epochs),
            batch=int(paths.train_batch),
            project=str(task.project_root),
            name="latest",
            exist_ok=True,
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
            device=str(paths.train_device),
        )

        best_weight = task.project_root / "latest" / "weights" / "best.pt"
        stable_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weight, stable_weights)

        with TRAINING_STATE.lock:
            TRAINING_STATE.status = "ready"
            TRAINING_STATE.message = f"{task.label} model trained from {len(samples)} annotated images. Generate predictions when ready."
            TRAINING_STATE.weights_path = repo_display_path(stable_weights)
            TRAINING_STATE.finished_at = time.time()
    except Exception as exc:
        with TRAINING_STATE.lock:
            TRAINING_STATE.status = "error"
            TRAINING_STATE.message = str(exc)
            TRAINING_STATE.error = str(exc)
            TRAINING_STATE.finished_at = time.time()
    finally:
        with TRAINING_STATE.lock:
            TRAINING_STATE.running = False


def schedule_prediction(paths: AppPaths, reason: str, task_key: str | None = None) -> dict:
    task = task_config(paths, task_key)
    with TRAINING_STATE.lock:
        if TRAINING_STATE.running:
            return {"scheduled": False, "reason": "AI job already running", "status": TRAINING_STATE.status}
        stable_weights = task.model_path
        if not stable_weights.exists():
            return {"scheduled": False, "reason": f"No trained model found for {task.label}. Train AI first."}
        TRAINING_STATE.running = True
        TRAINING_STATE.pending = False
        TRAINING_STATE.status = "queued"
        TRAINING_STATE.message = f"Queued {task.label} prediction generation after {reason}."
        TRAINING_STATE.error = None
    thread = threading.Thread(target=prediction_worker, args=(paths, task.key), daemon=True)
    thread.start()
    return {"scheduled": True, "pending": False, "task": task.to_json()}


def prediction_worker(paths: AppPaths, task_key: str) -> None:
    try:
        task = task_config(paths, task_key)
        stable_weights = task.model_path
        with TRAINING_STATE.lock:
            TRAINING_STATE.status = "predicting"
            TRAINING_STATE.message = f"Generating {task.label} AI predictions..."
            TRAINING_STATE.started_at = time.time()
            TRAINING_STATE.finished_at = None
            TRAINING_STATE.weights_path = repo_display_path(stable_weights)
            TRAINING_STATE.error = None

        predict_cmd = [
            sys.executable,
            "scripts/predict_unlabeled.py",
            "--weights",
            str(stable_weights),
            "--images-root",
            str(paths.images_root),
            "--labels-root",
            str(task.labels_root),
            "--output-root",
            str(task.predictions_root.parent),
            "--imgsz",
            str(paths.train_imgsz),
            "--conf",
            str(task.single_image_conf),
            "--include-labeled",
        ]
        if task.camera_filter:
            predict_cmd.extend(["--camera-filter", task.camera_filter])
        subprocess.run(predict_cmd, cwd=paths.root, check=True)

        with TRAINING_STATE.lock:
            TRAINING_STATE.status = "ready"
            TRAINING_STATE.message = f"{task.label} AI predictions generated."
            TRAINING_STATE.finished_at = time.time()
    except Exception as exc:
        with TRAINING_STATE.lock:
            TRAINING_STATE.status = "error"
            TRAINING_STATE.message = str(exc)
            TRAINING_STATE.error = str(exc)
            TRAINING_STATE.finished_at = time.time()
    finally:
        with TRAINING_STATE.lock:
            TRAINING_STATE.running = False


def _single_image_predict_conf() -> float:
    raw = os.environ.get("PIPE_END_ANNOTATOR_PREDICT_CONF", "0.20").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.20


def _single_image_predict_device() -> str | None:
    raw = os.environ.get("PIPE_END_YOLO_DEVICE", "").strip()
    return raw or None


def _fallback_model_for_task(task: AnnotationTask) -> Path:
    if task.model_path.exists():
        return task.model_path
    raise FileNotFoundError(f"No trained model found for {task.label}: {task.model_path}")


def _initial_training_model(task: AnnotationTask, base_model: str) -> str:
    if task.model_path.exists():
        return str(task.model_path)
    return str(base_model)


def _run_generic_yolo_prediction(
    *,
    paths: AppPaths,
    image_path: Path,
    rel: Path,
    output_dir: Path,
    task: AnnotationTask,
    imgsz: int,
    device: str | None,
) -> dict:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("Python package 'ultralytics' was not found. Install ultralytics first.") from exc

    model_path = _fallback_model_for_task(task)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = image_size(image_path)
    model = YOLO(str(model_path))
    predict_kwargs: dict[str, object] = {
        "imgsz": int(imgsz),
        "conf": float(task.single_image_conf),
        "verbose": False,
        "max_det": 16 if task.max_boxes else 256,
    }
    if device:
        predict_kwargs["device"] = str(device)
    results = model.predict(str(image_path), **predict_kwargs)
    boxes: list[dict] = []
    if results:
        result = results[0]
        raw_boxes = getattr(result, "boxes", None)
        if raw_boxes is not None and raw_boxes.xyxy is not None:
            xyxy = raw_boxes.xyxy.cpu().numpy()
            cls = raw_boxes.cls.cpu().numpy()
            scores = raw_boxes.conf.cpu().numpy()
            for class_id, coords, score in zip(cls, xyxy, scores):
                if int(class_id) != CLASS_ID:
                    continue
                x1, y1, x2, y2 = [float(value) for value in coords]
                boxes.append(
                    {
                        "class_id": CLASS_ID,
                        "x": max(0.0, min(float(width), x1)),
                        "y": max(0.0, min(float(height), y1)),
                        "w": max(0.0, min(float(width), x2) - max(0.0, min(float(width), x1))),
                        "h": max(0.0, min(float(height), y2) - max(0.0, min(float(height), y1))),
                        "conf": float(score),
                    }
                )
    boxes.sort(key=lambda box: float(box.get("conf") or 0.0), reverse=True)
    if task.max_boxes is not None:
        boxes = boxes[: task.max_boxes]
    prediction_label_path = (task.predictions_root / rel).with_suffix(".txt")
    prediction_label_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_label_path.write_text(boxes_to_yolo(boxes, width, height), encoding="utf-8")
    return {
        "ok": True,
        "boxes": boxes,
        "count": len(boxes),
        "width": width,
        "height": height,
        "model_path": repo_display_path(model_path),
        "prediction_path": repo_display_path(prediction_label_path),
        "overlay_path": None,
    }


def _sam_payload_y_bounds(payload: dict) -> tuple[float, float] | None:
    ys: list[float] = []
    for key in ("mask_polygon", "boundary", "left_boundary", "right_boundary"):
        points = payload.get(key)
        if not isinstance(points, list):
            continue
        for point in points:
            if isinstance(point, list) and len(point) >= 2:
                try:
                    ys.append(float(point[1]))
                except (TypeError, ValueError):
                    pass
    if len(ys) < 2:
        return None
    return min(ys), max(ys)


def _latest_sam_recovery_bounds_y(rel: Path) -> tuple[float, float] | None:
    boundaries_root = REPO_ROOT / "sam_boundary_detection" / "sam2p1_boundary_app" / "boundaries"
    candidates: list[tuple[float, float, float]] = []
    for source in ("sam_full_image", "pipe_end_span"):
        path = (boundaries_root / source / rel).with_suffix(".json")
        if not path.exists():
            continue
        try:
            bounds = _sam_payload_y_bounds(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            bounds = None
        if bounds is None:
            continue
        top, bottom = bounds
        candidates.append((bottom - top, top, bottom))
    if not candidates:
        return None
    _, top, bottom = max(candidates, key=lambda item: item[0])
    return float(top), float(bottom)


def run_single_image_prediction(paths: AppPaths, rel: Path, task_key: str | None = None) -> dict:
    from src.pipe_end_yolo import resolve_model_path, run_pipe_end_inference

    task = task_config(paths, task_key)
    image_path = paths.images_root / rel
    if not image_path.exists() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise FileNotFoundError(f"image not found: {rel.as_posix()}")
    if task.camera_filter and (not rel.parts or rel.parts[0] != task.camera_filter):
        raise ValueError(f"{task.label} only supports {task.camera_filter} images.")

    if not task.use_pipe_end_inference:
        output_dir = task.work_root / "single_image_runs" / rel.parent / image_path.stem
        return _run_generic_yolo_prediction(
            paths=paths,
            image_path=image_path,
            rel=rel,
            output_dir=output_dir,
            task=task,
            imgsz=paths.train_imgsz,
            device=_single_image_predict_device(),
        )

    model_path = _fallback_model_for_task(task)
    output_dir = task.work_root / "single_image_runs" / rel.parent / image_path.stem
    recovery_bounds_y = _latest_sam_recovery_bounds_y(rel)
    spacing_stats = _task_pipe_end_spacing_stats(paths, task)
    result = run_pipe_end_inference(
        image_path,
        output_dir,
        model_path=model_path,
        imgsz=paths.train_imgsz,
        conf=task.single_image_conf,
        device=_single_image_predict_device(),
        recovery_bounds_y=recovery_bounds_y,
        spacing_stats=spacing_stats,
    )

    prediction_label_path = (task.predictions_root / rel).with_suffix(".txt")
    prediction_label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for pred in result.predictions:
        x1 = max(0.0, min(float(result.image_width), float(pred.x1)))
        y1 = max(0.0, min(float(result.image_height), float(pred.y1)))
        x2 = max(0.0, min(float(result.image_width), float(pred.x2)))
        y2 = max(0.0, min(float(result.image_height), float(pred.y2)))
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0.0 or h <= 0.0:
            continue
        xc = (x1 + 0.5 * w) / float(result.image_width)
        yc = (y1 + 0.5 * h) / float(result.image_height)
        wn = w / float(result.image_width)
        hn = h / float(result.image_height)
        lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f} {float(pred.confidence):.6f}")
    prediction_label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    boxes = yolo_to_boxes(prediction_label_path, result.image_width, result.image_height)
    return {
        "ok": True,
        "boxes": boxes,
        "count": len(boxes),
        "width": result.image_width,
        "height": result.image_height,
        "model_path": repo_display_path(model_path),
        "prediction_path": repo_display_path(prediction_label_path),
        "overlay_path": repo_display_path(result.overlay_path),
        "recovery_bounds_y": None if recovery_bounds_y is None else [float(recovery_bounds_y[0]), float(recovery_bounds_y[1])],
        "spacing_stats": spacing_stats,
        "postprocess": result.postprocess or {},
    }


def _sam_boundary_model(model_name: str) -> object:
    with SAM_BOUNDARY_LOCK:
        model = SAM_BOUNDARY_MODELS.get(model_name)
        if model is not None:
            return model
        try:
            from ultralytics import SAM  # type: ignore
        except Exception as exc:
            raise RuntimeError("Python package 'ultralytics' was not found. Install ultralytics first.") from exc
        model = SAM(model_name)
        SAM_BOUNDARY_MODELS[model_name] = model
        return model


def _largest_component(mask: object) -> object:
    import cv2  # type: ignore
    import numpy as np

    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return mask_u8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _clean_sam_mask(mask: object, close_kernel: int = 7) -> object:
    import cv2  # type: ignore
    import numpy as np

    mask_u8 = np.asarray(_largest_component(mask), dtype=np.uint8)
    if close_kernel > 1:
        k = int(close_kernel)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_u8 = np.asarray(_largest_component(mask_u8), dtype=np.uint8)
    return mask_u8


def _crop_mask_to_box(mask: object, box: dict | None) -> object:
    import numpy as np

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if not isinstance(box, dict):
        return mask_u8
    height, width = mask_u8.shape[:2]
    try:
        x1 = int(max(0, min(width, round(float(box.get("x", 0.0))))))
        y1 = int(max(0, min(height, round(float(box.get("y", 0.0))))))
        x2 = int(max(0, min(width, round(float(box.get("x", 0.0)) + float(box.get("w", 0.0))))))
        y2 = int(max(0, min(height, round(float(box.get("y", 0.0)) + float(box.get("h", 0.0))))))
    except (TypeError, ValueError):
        return mask_u8
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return mask_u8
    cropped = np.zeros_like(mask_u8)
    cropped[y1:y2, x1:x2] = mask_u8[y1:y2, x1:x2]
    return cropped


def _mask_polygon(mask: object, epsilon_ratio: float = 0.003) -> list[list[float]]:
    import cv2  # type: ignore
    import numpy as np

    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, epsilon_ratio * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in approx.tolist()]


def _smooth_values(values: list[float], window: int) -> list[float]:
    import numpy as np

    if not values:
        return []
    k = max(1, int(window))
    if k <= 1:
        return [float(v) for v in values]
    if k % 2 == 0:
        k += 1
    pad = k // 2
    arr = np.asarray(values, dtype=np.float32)
    padded = np.pad(arr, (pad, pad), mode="edge")
    return [float(np.median(padded[idx : idx + k])) for idx in range(arr.size)]


def _mask_boundary(mask: object, side: str, row_step: int = 5, smooth_window: int = 9, min_width_px: int = 12) -> list[list[float]]:
    import numpy as np

    mask_arr = np.asarray(mask)
    height = int(mask_arr.shape[0])
    ys: list[int] = []
    xs: list[float] = []
    for y in range(0, height, max(1, int(row_step))):
        cols = np.flatnonzero(mask_arr[y] > 0)
        if cols.size < max(1, int(min_width_px)):
            continue
        ys.append(y)
        xs.append(float(cols[0] if side == "left" else cols[-1]))
    xs_smooth = _smooth_values(xs, smooth_window)
    return [[float(x), float(y)] for x, y in zip(xs_smooth, ys)]


def _boundary_roughness(points: list[list[float]]) -> float:
    import numpy as np

    if len(points) < 3:
        return 0.0
    xs = np.asarray([point[0] for point in points], dtype=np.float32)
    return float(np.std(np.diff(xs)))


def _sam_full_image_meta() -> dict:
    return {
        "prompt_source": "sam_full_image",
        "prompt_label_path": None,
        "prompt_model_path": None,
        "prompt_conf": None,
        "prompt_box": None,
    }


def _box_payload(box: dict) -> dict:
    return {
        "x": float(box.get("x", 0.0)),
        "y": float(box.get("y", 0.0)),
        "w": float(box.get("w", 0.0)),
        "h": float(box.get("h", 0.0)),
        "conf": box.get("conf"),
    }


def _box_center_y(box: dict) -> float:
    return float(box.get("y", 0.0)) + 0.5 * float(box.get("h", 0.0))


def _select_pipe_end_bundle_cluster(boxes: list[dict], spacing_stats: dict | None = None) -> list[dict]:
    if len(boxes) <= 2:
        return boxes
    ordered = sorted(boxes, key=_box_center_y)
    median_h = _median_float([float(box.get("h", 0.0)) for box in ordered]) or 16.0
    gaps = [
        _box_center_y(ordered[idx + 1]) - _box_center_y(ordered[idx])
        for idx in range(len(ordered) - 1)
        if _box_center_y(ordered[idx + 1]) > _box_center_y(ordered[idx])
    ]
    if spacing_stats:
        pitch = float(spacing_stats.get("median_gap_px") or 0.0)
        break_gap = float(spacing_stats.get("allowed_gap_px") or 0.0)
    else:
        pitch = 0.0
        break_gap = 0.0
    if pitch <= 1e-6 or break_gap <= 1e-6:
        local_gaps = [gap for gap in gaps if gap <= max(55.0, 4.0 * float(median_h))]
        pitch = _median_float(local_gaps) or _median_float(gaps) or (1.4 * float(median_h))
        break_gap = max(55.0, 2.8 * float(pitch), 3.5 * float(median_h))

    clusters: list[list[dict]] = [[ordered[0]]]
    previous_y = _box_center_y(ordered[0])
    for box in ordered[1:]:
        y = _box_center_y(box)
        if y - previous_y > break_gap:
            clusters.append([box])
        else:
            clusters[-1].append(box)
        previous_y = y

    kept_clusters = [cluster for cluster in clusters if len(cluster) >= 3]
    if not kept_clusters:
        kept_clusters = [max(clusters, key=len)]
    return [box for cluster in kept_clusters for box in cluster]


def _pipe_end_span_prompt(paths: AppPaths, rel: Path, width: int, height: int) -> tuple[tuple[float, float, float, float], dict]:
    camera = rel.parts[0] if rel.parts else ""
    task_key = "pipe_end_cam152" if camera == "cam152" else "pipe_end"
    task = task_config(paths, task_key)
    prediction = run_single_image_prediction(paths, rel, task_key)
    boxes = [
        box
        for box in prediction.get("boxes", [])
        if float(box.get("w", 0.0)) > 1.0 and float(box.get("h", 0.0)) > 1.0
    ]
    if not boxes:
        raise ValueError(f"{task.class_name} detector found 0 boxes for {rel.as_posix()}.")

    spacing_stats = _task_pipe_end_spacing_stats(paths, task)
    prompt_boxes = _select_pipe_end_bundle_cluster(boxes, spacing_stats)
    top = min(prompt_boxes, key=_box_center_y)
    bottom = max(prompt_boxes, key=_box_center_y)
    heights = sorted(float(box.get("h", 0.0)) for box in prompt_boxes)
    widths = sorted(float(box.get("w", 0.0)) for box in prompt_boxes)
    median_h = heights[len(heights) // 2] if heights else 0.0
    median_w = widths[len(widths) // 2] if widths else 0.0
    y_margin = max(10.0, 0.6 * median_h)
    top_y_margin = max(24.0, 2.0 * median_h)
    x_margin = max(12.0, 0.75 * median_w)
    strip_margin = max(48.0, 6.0 * median_w, 0.31 * float(width))
    left_edge = min(float(box.get("x", 0.0)) for box in prompt_boxes)
    right_edge = max(float(box.get("x", 0.0)) + float(box.get("w", 0.0)) for box in prompt_boxes)
    mask_crop_x1 = max(0.0, left_edge - strip_margin)
    x1 = 0.0
    y1 = max(1.0, float(top.get("y", 0.0)) - top_y_margin)
    x2 = min(float(width), right_edge + x_margin)
    y2 = min(float(height), float(bottom.get("y", 0.0)) + float(bottom.get("h", 0.0)) + y_margin)
    if x2 <= x1 + 2.0 or y2 <= y1 + 2.0:
        raise ValueError(f"pipe_end span prompt was too small for {rel.as_posix()}.")

    top_payload = _box_payload(top)
    bottom_payload = _box_payload(bottom)
    conf_values = [
        float(box.get("conf"))
        for box in (top, bottom)
        if box.get("conf") is not None
    ]
    prompt_conf = min(conf_values) if conf_values else None
    return (x1, y1, x2, y2), {
        "prompt_source": "pipe_end_span",
        "prompt_label_path": repo_display_path((task.predictions_root / rel).with_suffix(".txt")),
        "prompt_model_path": prediction.get("model_path"),
        "prompt_conf": prompt_conf,
        "prompt_pipe_end_count": len(prompt_boxes),
        "prompt_pipe_end_raw_count": len(boxes),
        "prompt_spacing_stats": spacing_stats,
        "prompt_pipe_end_boxes": [top_payload, bottom_payload],
        "prompt_top_bound_source": "highest_pipe_end_expanded",
        "prompt_bottom_bound_source": "lowest_pipe_end",
        "prompt_x_bound_source": "wide_context",
        "mask_crop_box": {
            "x": mask_crop_x1,
            "y": y1,
            "w": x2 - mask_crop_x1,
            "h": y2 - y1,
            "source": "pipe_end_strip_cleanup",
        },
    }


def run_sam_boundary(
    paths: AppPaths,
    rel: Path,
    side: str = "right",
    sam_model: str = "sam2.1_s.pt",
    prompt_mode: str = "pipe_end_span",
) -> dict:
    import cv2  # type: ignore
    import numpy as np

    image_path = paths.images_root / rel
    if not image_path.exists() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise FileNotFoundError(f"image not found: {rel.as_posix()}")

    width, height = image_size(image_path)
    prompt_mode = (prompt_mode or "pipe_end_span").strip().lower()
    prompt_box: tuple[float, float, float, float] | None = None
    if prompt_mode == "full_image":
        prompt_meta = _sam_full_image_meta()
    elif prompt_mode in {"pipe_end_span", "pipe_ends", "pipe_end"}:
        prompt_box, prompt_meta = _pipe_end_span_prompt(paths, rel, width, height)
    else:
        raise ValueError(f"Unknown SAM prompt mode: {prompt_mode!r}")

    model = _sam_boundary_model(sam_model)
    predict_kwargs: dict[str, object] = {
        "verbose": False,
    }
    if prompt_box is not None:
        predict_kwargs["bboxes"] = [list(prompt_box)]
    if paths.train_device:
        predict_kwargs["device"] = str(paths.train_device)
    results = model.predict(str(image_path), **predict_kwargs)  # type: ignore[attr-defined]
    if not results or getattr(results[0], "masks", None) is None:
        raise RuntimeError("SAM did not return a mask.")
    data = results[0].masks.data
    if data is None or len(data) == 0:
        raise RuntimeError("SAM returned an empty mask.")
    masks = data.cpu().numpy().astype(np.uint8)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    mask_areas = [int(mask.sum()) for mask in masks]
    selected_mask_index = int(np.argmax(mask_areas))
    mask = masks[selected_mask_index]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = np.asarray(_clean_sam_mask(mask), dtype=np.uint8)
    mask_crop_box = prompt_meta.get("mask_crop_box") if isinstance(prompt_meta, dict) else None
    if mask_crop_box is not None:
        cropped_mask = np.asarray(_crop_mask_to_box(mask, mask_crop_box), dtype=np.uint8)
        if int(cropped_mask.sum()) > 0:
            mask = np.asarray(_clean_sam_mask(cropped_mask, close_kernel=3), dtype=np.uint8)

    left = _mask_boundary(mask, "left")
    right = _mask_boundary(mask, "right")
    if side not in {"left", "right", "auto"}:
        side = "right"
    selected_side = side
    if selected_side == "auto":
        selected_side = "left" if _boundary_roughness(left) > _boundary_roughness(right) else "right"
    boundary = left if selected_side == "left" else right
    polygon = _mask_polygon(mask)

    output_path = (
        REPO_ROOT
        / "sam_boundary_detection"
        / "sam2p1_boundary_app"
        / "boundaries"
        / str(prompt_meta.get("prompt_source", "unknown"))
        / rel
    ).with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "path": rel.as_posix(),
        "width": width,
        "height": height,
        "sam_model": sam_model,
        **prompt_meta,
        "prompt_box": (
            {"x": prompt_box[0], "y": prompt_box[1], "w": prompt_box[2] - prompt_box[0], "h": prompt_box[3] - prompt_box[1]}
            if prompt_box is not None
            else None
        ),
        "mask_count": int(len(masks)),
        "selected_mask_index": selected_mask_index,
        "selected_side": selected_side,
        "boundary": boundary,
        "left_boundary": left,
        "right_boundary": right,
        "mask_polygon": polygon,
        "mask_area_px": int(mask.sum()),
        "left_roughness": _boundary_roughness(left),
        "right_roughness": _boundary_roughness(right),
        "boundary_path": repo_display_path(output_path),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _read_roi_toml(path: Path) -> dict:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        "src_points_override": data.get("src_points_override"),
        "dst_rect_override": data.get("dst_rect_override"),
    }


def _save_roi_dst_rect(path: Path, new_rect: list[float]) -> None:
    text = path.read_text(encoding="utf-8")
    formatted = ", ".join(f"{v:.3f}" for v in new_rect)
    new_line = f"dst_rect_override = [{formatted}]"
    if re.search(r"^dst_rect_override\s*=", text, re.MULTILINE):
        text = re.sub(r"^dst_rect_override\s*=.*$", new_line, text, flags=re.MULTILINE)
    else:
        text = text.rstrip() + f"\n{new_line}\n"
    path.write_text(text, encoding="utf-8")


def _build_full_warp_jpeg(raw_image_path: Path, src_points: list) -> tuple[bytes, list[float], tuple[int, int]]:
    """Warp the full raw image without dst_rect crop. Returns (jpeg_bytes, view_rect, (w, h))."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    img = cv2.imread(str(raw_image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {raw_image_path}")
    ih, iw = img.shape[:2]

    pts = [(float(p[0]), float(p[1])) for p in src_points]
    tl, tr, bl, br = pts

    def _d(a: tuple, b: tuple) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    bw = max(240, int(round(max(_d(tl, tr), _d(bl, br)))))
    bh = max(320, int(round(max(_d(tl, bl), _d(tr, br)))))

    src = np.float32(pts)
    dst = np.float32([[0, 0], [bw - 1, 0], [0, bh - 1], [bw - 1, bh - 1]])
    M = cv2.getPerspectiveTransform(src, dst)

    corners = np.float32([[[0, 0]], [[iw - 1, 0]], [[iw - 1, ih - 1]], [[0, ih - 1]]])
    proj = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    min_x = float(np.floor(proj[:, 0].min()))
    min_y = float(np.floor(proj[:, 1].min()))
    max_x = float(np.ceil(proj[:, 0].max()))
    max_y = float(np.ceil(proj[:, 1].max()))
    ow = max(80, int(round(max_x - min_x)))
    oh = max(80, int(round(max_y - min_y)))

    T = np.float64([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    warped = cv2.warpPerspective(img, T @ M.astype(np.float64), (ow, oh))
    ok, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("No se pudo codificar el warp como JPEG")
    return bytes(buf), [min_x, min_y, max_x, max_y], (ow, oh)


def _roi_toml_for_camera(paths: "AppPaths", camera: str) -> "Path | None":
    side = "151" if camera in ("cam151", "151") else "152" if camera in ("cam152", "152") else None
    if side is None:
        return None
    return paths.roi_toml_151 if side == "151" else paths.roi_toml_152


def _raw_image_for_camera(paths: "AppPaths", camera: str) -> "Path | None":
    side = "151" if camera in ("cam151", "151") else "152"
    return paths.raw_image_151 if side == "151" else paths.raw_image_152


ROI_PICKER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ROI Picker — Pipe End</title>
  <style>
    :root {
      --bg: #101214; --panel: #1a1d21; --panel2: #22262c;
      --line: #343a43; --text: #f2f1eb; --muted: #a8afba;
      --accent: #ffd43b; --accent2: #32d583; --danger: #ff5c5c;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text);
           font-family: "Segoe UI", Tahoma, sans-serif; overflow: hidden; }
    .app { display: grid; grid-template-columns: 300px 1fr; height: 100vh; }
    aside { background: linear-gradient(180deg, var(--panel), #111315);
            border-right: 1px solid var(--line);
            display: flex; flex-direction: column; overflow: hidden; }
    header { padding: 16px; border-bottom: 1px solid var(--line); flex: 0 0 auto; }
    h1 { margin: 0 0 6px; font-size: 18px; }
    .small { color: var(--muted); font-size: 12px; line-height: 1.4; }
    .controls { padding: 12px; border-bottom: 1px solid var(--line);
                display: grid; gap: 8px; flex: 0 0 auto; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    select, button { border: 1px solid var(--line); border-radius: 8px;
                     background: var(--panel2); color: var(--text);
                     padding: 9px 10px; font-size: 14px; }
    button { cursor: pointer; font-weight: 650; }
    button:hover { border-color: var(--accent); }
    button:disabled { opacity: 0.4; cursor: default; }
    .primary { background: #5b4405; border-color: #8d6b0f; color: #fff7d1; }
    .info { padding: 12px; font-size: 12px; color: var(--muted);
            border-bottom: 1px solid var(--line); flex: 0 0 auto; }
    .info b { color: var(--text); }
    .info-row { margin-bottom: 5px; }
    .log { padding: 12px; font-size: 12px; color: var(--muted);
           flex: 1; overflow-y: auto; white-space: pre-wrap; }
    main { display: grid; grid-template-rows: auto 1fr auto; min-width: 0; min-height: 0; }
    .toolbar { display: flex; align-items: center; gap: 8px; padding: 8px 12px;
               border-bottom: 1px solid var(--line); background: rgba(0,0,0,.2); }
    .toolbar .title { flex: 1; font-weight: 700; font-size: 13px;
                      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .stage { overflow: hidden; position: relative; touch-action: none; background: #060708; }
    .canvas-wrap { position: absolute; left: 0; top: 0; transform-origin: top left;
                   will-change: transform; background: #060708; line-height: 0; }
    #image { display: block; user-select: none; pointer-events: none; }
    #overlay { position: absolute; inset: 0; cursor: crosshair; }
    .footer { display: flex; gap: 12px; align-items: center; padding: 8px 12px;
               border-top: 1px solid var(--line); color: var(--muted); font-size: 12px;
               background: rgba(0,0,0,.25); flex: 0 0 auto; }
    .kbd { border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px;
           padding: 1px 5px; background: var(--panel2); font-size: 11px; color: var(--text); }
    #statusSpan { font-weight: 700; color: var(--accent); }
  </style>
</head>
<body>
<div class="app">
  <aside>
    <header>
      <h1>ROI Picker</h1>
      <div class="small">Dibuja un rectángulo sobre la imagen warpeada completa para definir la región de recorte. El rectángulo verde es el nuevo ROI; el amarillo punteado es el actual.</div>
    </header>
    <div class="controls">
      <select id="camSel"><option value="cam151">cam151</option><option value="cam152">cam152</option></select>
      <button id="loadBtn">Cargar imagen</button>
      <div class="row">
        <button id="saveBtn" class="primary" disabled>Guardar (Ctrl+S)</button>
        <button id="resetBtn">Resetear</button>
      </div>
    </div>
    <div class="info" id="infoPanel">
      <div class="info-row"><b>ROI actual:</b><br><span id="curRoi">—</span></div>
      <div class="info-row" style="margin-top:8px"><b>ROI nuevo:</b><br><span id="newRoi">—</span></div>
      <div class="info-row"><b>Tamaño nuevo:</b> <span id="newSize">—</span></div>
    </div>
    <div class="log" id="logBox">Selecciona cámara y haz click en "Cargar imagen".</div>
  </aside>
  <main>
    <div class="toolbar">
      <div class="title" id="toolTitle">—</div>
      <button id="zoomOutBtn">−</button>
      <button id="fitBtn">Fit</button>
      <button id="zoomInBtn">+</button>
    </div>
    <div class="stage" id="stage">
      <div class="canvas-wrap" id="wrap">
        <img id="image" alt="" />
        <canvas id="overlay"></canvas>
      </div>
    </div>
    <div class="footer">
      <span id="statusSpan">listo</span>
      <span><span class="kbd">drag</span> dibujar rect</span>
      <span><span class="kbd">drag interior</span> mover</span>
      <span><span class="kbd">drag esquina</span> redimensionar</span>
      <span><span class="kbd">wheel</span> zoom</span>
      <span><span class="kbd">Space+drag</span> pan</span>
      <span><span class="kbd">Ctrl+S</span> guardar</span>
    </div>
  </main>
</div>
<script>
const S = {
  zoom:1, panX:18, panY:18,
  imgW:0, imgH:0,
  viewRect: null,   // [min_x, min_y, max_x, max_y] homography offset
  curRect: null,    // [x0,y0,x1,y1] homo space — current dst_rect_override
  newRect: null,    // {x,y,w,h} pixel space — user selection
  drawing: null,    // {sx,sy,ex,ey}
  dragging: null,   // {mode,handle,sx,sy,orig}
  panning: null,
  spaceDown: false,
  camera: 'cam151',
};

const E = {
  camSel: document.getElementById('camSel'),
  loadBtn: document.getElementById('loadBtn'),
  saveBtn: document.getElementById('saveBtn'),
  resetBtn: document.getElementById('resetBtn'),
  image: document.getElementById('image'),
  overlay: document.getElementById('overlay'),
  wrap: document.getElementById('wrap'),
  stage: document.getElementById('stage'),
  toolTitle: document.getElementById('toolTitle'),
  curRoi: document.getElementById('curRoi'),
  newRoi: document.getElementById('newRoi'),
  newSize: document.getElementById('newSize'),
  logBox: document.getElementById('logBox'),
  statusSpan: document.getElementById('statusSpan'),
  zoomInBtn: document.getElementById('zoomInBtn'),
  zoomOutBtn: document.getElementById('zoomOutBtn'),
  fitBtn: document.getElementById('fitBtn'),
};
const ctx = E.overlay.getContext('2d');

function log(msg) { E.logBox.textContent = msg; }
function status(msg) { E.statusSpan.textContent = msg; }

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text() || r.statusText);
  return r.json();
}

function applyZoom() {
  E.wrap.style.transform = `translate(${S.panX}px,${S.panY}px) scale(${S.zoom})`;
  status(`zoom ${(S.zoom*100).toFixed(0)}%`);
}

function fitZoom() {
  if (!S.imgW) return;
  const pad = 40;
  const zw = (E.stage.clientWidth - pad) / S.imgW;
  const zh = (E.stage.clientHeight - pad) / S.imgH;
  S.zoom = Math.max(0.05, Math.min(1.5, zw, zh));
  S.panX = Math.max(18, (E.stage.clientWidth - S.imgW * S.zoom) / 2);
  S.panY = Math.max(18, (E.stage.clientHeight - S.imgH * S.zoom) / 2);
  applyZoom();
}

function zoomAt(cx, cy, f) {
  const r = E.stage.getBoundingClientRect();
  const lx = cx - r.left, ly = cy - r.top;
  const ix = (lx - S.panX) / S.zoom, iy = (ly - S.panY) / S.zoom;
  S.zoom = Math.max(0.05, Math.min(8, S.zoom * f));
  S.panX = lx - ix * S.zoom; S.panY = ly - iy * S.zoom;
  applyZoom();
}

function canvasPt(evt) {
  const r = E.stage.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(S.imgW, (evt.clientX - r.left - S.panX) / S.zoom)),
    y: Math.max(0, Math.min(S.imgH, (evt.clientY - r.top  - S.panY) / S.zoom)),
  };
}

// pixel → homography space
function p2h(px, py) {
  return S.viewRect ? [S.viewRect[0]+px, S.viewRect[1]+py] : [px, py];
}

// homography → pixel space
function h2p(hx, hy) {
  return S.viewRect ? [hx-S.viewRect[0], hy-S.viewRect[1]] : [hx, hy];
}

function curRectPx() {
  if (!S.curRect || !S.viewRect) return null;
  const [px0, py0] = h2p(S.curRect[0], S.curRect[1]);
  const [px1, py1] = h2p(S.curRect[2], S.curRect[3]);
  return {x:px0, y:py0, w:px1-px0, h:py1-py0};
}

function normDraw() {
  if (!S.drawing) return null;
  const {sx,sy,ex,ey} = S.drawing;
  return {x:Math.min(sx,ex), y:Math.min(sy,ey), w:Math.abs(ex-sx), h:Math.abs(ey-sy)};
}

// 8 handles: NW N NE E SE S SW W
const HDIRS = ['nw','n','ne','e','se','s','sw','w'];
function handles(r) {
  const cx=r.x+r.w/2, cy=r.y+r.h/2;
  return [{x:r.x,y:r.y},{x:cx,y:r.y},{x:r.x+r.w,y:r.y},{x:r.x+r.w,y:cy},
          {x:r.x+r.w,y:r.y+r.h},{x:cx,y:r.y+r.h},{x:r.x,y:r.y+r.h},{x:r.x,y:cy}];
}
const HCURSOR = ['nw-resize','n-resize','ne-resize','e-resize','se-resize','s-resize','sw-resize','w-resize'];

function hitHandle(pt, r) {
  const hr = Math.max(5, 7/S.zoom);
  const hs = handles(r);
  for (let i=0;i<hs.length;i++) {
    const dx=pt.x-hs[i].x, dy=pt.y-hs[i].y;
    if (dx*dx+dy*dy < hr*hr) return i;
  }
  return -1;
}

function hitInside(pt, r) {
  return pt.x>r.x && pt.x<r.x+r.w && pt.y>r.y && pt.y<r.y+r.h;
}

function applyHandle(orig, hi, dx, dy) {
  let {x,y,w,h} = orig;
  const d = HDIRS[hi];
  if (d.includes('w')) { x+=dx; w-=dx; }
  if (d.includes('e')) { w+=dx; }
  if (d.includes('n')) { y+=dy; h-=dy; }
  if (d.includes('s')) { h+=dy; }
  if (w<0) { x+=w; w=-w; }
  if (h<0) { y+=h; h=-h; }
  x=Math.max(0,x); y=Math.max(0,y);
  w=Math.min(S.imgW-x,w); h=Math.min(S.imgH-y,h);
  return {x,y,w,h};
}

function updateInfoPanel(nr) {
  if (!nr) { E.newRoi.textContent='—'; E.newSize.textContent='—'; E.saveBtn.disabled=true; return; }
  const [hx0,hy0]=p2h(nr.x,nr.y), [hx1,hy1]=p2h(nr.x+nr.w,nr.y+nr.h);
  E.newRoi.textContent=`[${hx0.toFixed(1)}, ${hy0.toFixed(1)}, ${hx1.toFixed(1)}, ${hy1.toFixed(1)}]`;
  E.newSize.textContent=`${Math.round(nr.w)} × ${Math.round(nr.h)} px`;
  E.saveBtn.disabled=false;
}

function draw() {
  ctx.clearRect(0,0,E.overlay.width,E.overlay.height);

  // Current ROI — yellow dashed
  const cur = curRectPx();
  if (cur) {
    ctx.save();
    ctx.strokeStyle='rgba(255,212,59,0.55)'; ctx.lineWidth=2; ctx.setLineDash([8,5]);
    ctx.fillStyle='rgba(255,212,59,0.07)';
    ctx.fillRect(cur.x,cur.y,cur.w,cur.h);
    ctx.strokeRect(cur.x,cur.y,cur.w,cur.h);
    ctx.restore();
  }

  // New rect — green solid
  const nr = S.newRect || normDraw();
  if (nr) {
    ctx.save();
    ctx.strokeStyle='#32d583'; ctx.lineWidth=2.5; ctx.setLineDash([]);
    ctx.fillStyle='rgba(50,213,131,0.12)';
    ctx.fillRect(nr.x,nr.y,nr.w,nr.h);
    ctx.strokeRect(nr.x,nr.y,nr.w,nr.h);
    if (S.newRect) {
      const hs=handles(nr);
      const hr=Math.max(4,6/S.zoom);
      ctx.fillStyle='#32d583'; ctx.strokeStyle='#101214'; ctx.lineWidth=1.5;
      hs.forEach(h => { ctx.beginPath(); ctx.arc(h.x,h.y,hr,0,Math.PI*2); ctx.fill(); ctx.stroke(); });
    }
    ctx.restore();
    updateInfoPanel(nr);
  } else {
    updateInfoPanel(null);
  }
}

async function load() {
  S.camera = E.camSel.value;
  log('Cargando…');
  try {
    const info = await api(`/api/roi-info?camera=${S.camera}`);
    if (info.error) throw new Error(info.error);
    S.viewRect = info.view_rect;
    S.curRect  = info.dst_rect;
    S.newRect  = null; S.drawing = null;

    if (S.curRect) {
      const [x0,y0,x1,y1]=S.curRect;
      E.curRoi.textContent=`[${x0.toFixed(1)}, ${y0.toFixed(1)}, ${x1.toFixed(1)}, ${y1.toFixed(1)}]`;
    }
    updateInfoPanel(null);

    E.image.onload = () => {
      S.imgW=E.image.naturalWidth; S.imgH=E.image.naturalHeight;
      E.overlay.width=S.imgW; E.overlay.height=S.imgH;
      E.wrap.style.width=S.imgW+'px'; E.wrap.style.height=S.imgH+'px';
      E.image.style.width=S.imgW+'px'; E.image.style.height=S.imgH+'px';
      E.overlay.style.width=S.imgW+'px'; E.overlay.style.height=S.imgH+'px';
      fitZoom(); draw();
      const src = info.source_type === 'raw' ? 'warp completo de imagen raw' : 'imagen de anotación (sin imagen raw configurada)';
      E.toolTitle.textContent=`${S.camera} — ${S.imgW}×${S.imgH}px`;
      log(`Cargado: ${src}.\nDibuja un rectángulo verde para definir el nuevo ROI.\nEl contorno amarillo es el ROI actual.`);
    };
    E.image.src=`/roi-warp-image?camera=${S.camera}&t=${Date.now()}`;
  } catch(e) { log('Error: '+e.message); }
}

async function save() {
  const nr=S.newRect; if (!nr) return;
  const [hx0,hy0]=p2h(nr.x,nr.y), [hx1,hy1]=p2h(nr.x+nr.w,nr.y+nr.h);
  try {
    await api('/api/roi-save',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({camera:S.camera,dst_rect:[hx0,hy0,hx1,hy1]})});
    S.curRect=[hx0,hy0,hx1,hy1];
    E.curRoi.textContent=`[${hx0.toFixed(1)}, ${hy0.toFixed(1)}, ${hx1.toFixed(1)}, ${hy1.toFixed(1)}]`;
    S.newRect=null; draw();
    log(`✓ Guardado en TOML.\nNuevo dst_rect_override = [${hx0.toFixed(3)}, ${hy0.toFixed(3)}, ${hx1.toFixed(3)}, ${hy1.toFixed(3)}]`);
  } catch(e) { log('Error al guardar: '+e.message); }
}

// ── Mouse events ──────────────────────────────────────────────────────────────
E.overlay.addEventListener('mousedown', evt => {
  if (evt.button===1||evt.button===2||S.spaceDown) {
    S.panning={x:evt.clientX,y:evt.clientY,px:S.panX,py:S.panY};
    E.overlay.style.cursor='grabbing'; evt.preventDefault(); return;
  }
  if (evt.button!==0) return;
  const pt=canvasPt(evt);
  if (S.newRect) {
    const hi=hitHandle(pt,S.newRect);
    if (hi>=0) { S.dragging={mode:'handle',handle:hi,sx:pt.x,sy:pt.y,orig:{...S.newRect}}; return; }
    if (hitInside(pt,S.newRect)) { S.dragging={mode:'move',sx:pt.x,sy:pt.y,orig:{...S.newRect}}; E.overlay.style.cursor='move'; return; }
  }
  S.drawing={sx:pt.x,sy:pt.y,ex:pt.x,ey:pt.y}; S.newRect=null; draw();
});

E.stage.addEventListener('mousedown', evt => {
  if (evt.button===1||S.spaceDown) {
    S.panning={x:evt.clientX,y:evt.clientY,px:S.panX,py:S.panY};
    E.stage.style.cursor='grabbing'; evt.preventDefault();
  }
});

window.addEventListener('mousemove', evt => {
  if (S.panning) {
    S.panX=S.panning.px+(evt.clientX-S.panning.x);
    S.panY=S.panning.py+(evt.clientY-S.panning.y);
    applyZoom(); return;
  }
  if (S.dragging) {
    const pt=canvasPt(evt);
    const dx=pt.x-S.dragging.sx, dy=pt.y-S.dragging.sy;
    const o=S.dragging.orig;
    S.newRect = S.dragging.mode==='move'
      ? {x:Math.max(0,Math.min(S.imgW-o.w,o.x+dx)),y:Math.max(0,Math.min(S.imgH-o.h,o.y+dy)),w:o.w,h:o.h}
      : applyHandle(o,S.dragging.handle,dx,dy);
    draw(); return;
  }
  if (S.drawing) { const pt=canvasPt(evt); S.drawing.ex=pt.x; S.drawing.ey=pt.y; draw(); return; }
  // cursor hints
  if (S.newRect && !S.spaceDown) {
    const pt=canvasPt(evt);
    const hi=hitHandle(pt,S.newRect);
    if (hi>=0) { E.overlay.style.cursor=HCURSOR[hi]; return; }
    if (hitInside(pt,S.newRect)) { E.overlay.style.cursor='move'; return; }
  }
  if (!S.panning) E.overlay.style.cursor=S.spaceDown?'grab':'crosshair';
});

window.addEventListener('mouseup', () => {
  if (S.panning) { S.panning=null; E.overlay.style.cursor='crosshair'; E.stage.style.cursor='default'; return; }
  if (S.dragging) { S.dragging=null; E.overlay.style.cursor='crosshair'; return; }
  if (S.drawing) {
    const nr=normDraw(); S.drawing=null;
    if (nr&&nr.w>=5&&nr.h>=5) S.newRect=nr;
    draw();
  }
});

E.overlay.addEventListener('contextmenu', e=>e.preventDefault());
E.overlay.addEventListener('auxclick', e=>{if(e.button===1)e.preventDefault();});
E.stage.addEventListener('contextmenu', e=>e.preventDefault());
E.stage.addEventListener('wheel', evt=>{
  evt.preventDefault();
  zoomAt(evt.clientX,evt.clientY,evt.deltaY<0?1.12:1/1.12);
},{passive:false});

window.addEventListener('keydown', evt=>{
  const typing=['input','select','textarea'].includes(document.activeElement?.tagName?.toLowerCase());
  if (evt.code==='Space'&&!typing) { evt.preventDefault(); S.spaceDown=true; E.overlay.style.cursor='grab'; }
  if (evt.ctrlKey&&evt.key==='s') { evt.preventDefault(); save(); }
  if (evt.key==='Escape') { S.newRect=null; S.drawing=null; draw(); }
});
window.addEventListener('keyup', evt=>{
  if (evt.code==='Space') { S.spaceDown=false; if(!S.panning) E.overlay.style.cursor='crosshair'; }
});

E.loadBtn.onclick=load;
E.saveBtn.onclick=save;
E.resetBtn.onclick=()=>{ S.newRect=null; S.drawing=null; draw(); };
E.zoomInBtn.onclick=()=>{ const r=E.stage.getBoundingClientRect(); zoomAt(r.left+r.width/2,r.top+r.height/2,1.2); };
E.zoomOutBtn.onclick=()=>{ const r=E.stage.getBoundingClientRect(); zoomAt(r.left+r.width/2,r.top+r.height/2,1/1.2); };
E.fitBtn.onclick=fitZoom;

load();
</script>
</body>
</html>
"""


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pipe End Annotator</title>
  <style>
    :root {
      --bg: #101214;
      --panel: #1a1d21;
      --panel2: #22262c;
      --line: #343a43;
      --text: #f2f1eb;
      --muted: #a8afba;
      --accent: #ffd43b;
      --accent2: #32d583;
      --danger: #ff5c5c;
      --blue: #73c0ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top left, #26313a 0, var(--bg) 38%);
      color: var(--text);
      font-family: "Segoe UI", Tahoma, sans-serif;
      overflow: hidden;
    }
    .app {
      display: grid;
      grid-template-columns: 320px 1fr;
      height: 100vh;
    }
    aside {
      background: linear-gradient(180deg, var(--panel), #111315);
      border-right: 1px solid var(--line);
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    header {
      padding: 16px;
      border-bottom: 1px solid var(--line);
      flex: 0 0 auto;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.1;
    }
    .small { color: var(--muted); font-size: 12px; line-height: 1.35; }
    .controls {
      display: grid;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.02);
      flex: 0 0 auto;
    }
    select, input, button {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel2);
      color: var(--text);
      padding: 9px 10px;
      font-size: 14px;
    }
    button {
      cursor: pointer;
      font-weight: 650;
    }
    button:hover { border-color: var(--accent); }
    .primary { background: #5b4405; border-color: #8d6b0f; color: #fff7d1; }
    .danger { background: #461719; border-color: #8a2d31; color: #ffdede; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .list {
      overflow-y: auto;
      overflow-x: hidden;
      padding: 8px;
      flex: 1;
      min-height: 0;
      scrollbar-gutter: stable;
      overscroll-behavior: contain;
    }
    .item {
      border: 1px solid transparent;
      border-radius: 10px;
      padding: 8px 10px;
      margin-bottom: 5px;
      cursor: pointer;
      color: var(--muted);
      background: rgba(255,255,255,0.025);
      font-size: 12px;
    }
    .item:hover { border-color: #4d5663; }
    .item.active {
      border-color: var(--accent);
      background: rgba(255, 212, 59, 0.12);
      color: var(--text);
    }
    .item.done .badge { background: rgba(50,213,131,0.16); color: #a7f3c7; }
    .item.negative {
      border-color: rgba(115, 192, 255, 0.45);
      background: rgba(115, 192, 255, 0.10);
    }
    .item.negative .badge { background: rgba(115,192,255,0.18); color: #d8efff; }
    .item.bad {
      border-color: rgba(255, 92, 92, 0.45);
      background: rgba(255, 92, 92, 0.10);
    }
    .item.bad .badge { background: rgba(255,92,92,0.18); color: #ffdede; }
    .item-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .badge {
      display: inline-block;
      margin-top: 5px;
      padding: 2px 7px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      color: var(--muted);
      font-size: 11px;
    }
    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
      min-height: 0;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(0,0,0,0.18);
      min-width: 0;
    }
    .toolbar .title {
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
    }
    .stage {
      min-height: 0;
      overflow: hidden;
      display: block;
      padding: 18px;
      position: relative;
      touch-action: none;
    }
    .canvas-wrap {
      position: absolute;
      left: 0;
      top: 0;
      box-shadow: 0 18px 60px rgba(0,0,0,0.5);
      background: #060708;
      line-height: 0;
      transform-origin: top left;
      margin: 0;
      will-change: transform;
    }
    #image {
      display: block;
      user-select: none;
      pointer-events: none;
      max-width: none;
    }
    #overlay {
      position: absolute;
      inset: 0;
      cursor: crosshair;
    }
    .footer {
      display: flex;
      gap: 14px;
      align-items: center;
      padding: 9px 12px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      background: rgba(0,0,0,0.25);
    }
    .kbd {
      border: 1px solid var(--line);
      border-bottom-width: 2px;
      border-radius: 5px;
      padding: 1px 5px;
      color: var(--text);
      background: var(--panel2);
      font-size: 11px;
    }
    .dirty { color: var(--accent); font-weight: 700; }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; grid-template-rows: 260px 1fr; }
      aside { border-right: none; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>Tube Annotator</h1>
        <div class="small" id="taskHelp">Choose a task, then draw boxes for class <b>0</b>.</div>
      </header>
      <div class="controls">
        <select id="taskSelect">
          <option value="pipe_end">pipe_end | cam151 model</option>
          <option value="pipe_end_cam152">pipe_end | cam152 model</option>
        </select>
        <div class="row">
          <select id="cameraFilter">
            <option value="all">all cameras</option>
            <option value="cam151">cam151</option>
            <option value="cam152">cam152</option>
          </select>
          <select id="statusFilter">
            <option value="all">all</option>
            <option value="todo">todo</option>
            <option value="done">done</option>
            <option value="negative">negative</option>
            <option value="bad">bad warp</option>
            <option value="usable">usable only</option>
          </select>
        </div>
        <input id="search" placeholder="search timestamp..." />
        <div class="row">
          <button id="prevBtn">Prev</button>
          <button id="nextBtn">Next</button>
        </div>
        <div class="row">
          <button class="primary" id="saveBtn">Save</button>
          <button id="negativeBtn">Negative image</button>
        </div>
        <div class="row">
          <button class="danger" id="deleteBtn">Delete box</button>
          <button class="danger" id="badWarpBtn">Bad warp</button>
        </div>
        <div class="row">
          <button id="runImagePredBtn">Run model on this image</button>
          <button id="loadPredBtn">Load AI boxes</button>
        </div>
        <div class="row">
          <button id="runPipeEndSamBtn">Pipe-end SAM</button>
          <button id="runSamBoundaryBtn">Full-image SAM</button>
        </div>
        <button id="clearSamBoundaryBtn">Clear SAM</button>
        <div class="row">
          <button id="trainBtn">Train AI</button>
          <button id="generatePredBtn">Generate predictions</button>
        </div>
      </div>
      <div class="list" id="imageList"></div>
    </aside>
    <main>
      <div class="toolbar">
        <div class="title" id="title">Loading...</div>
        <button id="zoomOutBtn">-</button>
        <button id="fitBtn">Fit</button>
        <button id="zoomInBtn">+</button>
        <button id="modeBtn">Mode: Draw</button>
        <button id="clearBtn" class="danger">Clear</button>
      </div>
      <div class="stage" id="stage">
        <div class="canvas-wrap" id="wrap">
          <img id="image" alt="" />
          <canvas id="overlay"></canvas>
        </div>
      </div>
      <div class="footer">
        <span id="status">ready</span>
        <span id="trainStatus">AI idle</span>
        <span><span class="kbd">drag</span> draw box</span>
        <span><span class="kbd">wheel</span> zoom</span>
        <span><span class="kbd">middle/right drag</span> pan</span>
        <span><span class="kbd">Space+drag</span> pan</span>
        <span><span class="kbd">click</span> select</span>
        <span><span class="kbd">drag handle</span> resize</span>
        <span><span class="kbd">Del</span> delete</span>
        <span><span class="kbd">Ctrl+S</span> save</span>
        <span><span class="kbd">←/→</span> prev/next</span>
      </div>
    </main>
  </div>

  <script>
    const state = {
      task: 'pipe_end',
      tasks: {},
      images: [],
      filtered: [],
      currentIndex: 0,
      boxes: [],
      selected: -1,
      drawing: null,
      resizing: null,
      panning: null,
      negative: false,
      spaceDown: false,
      mode: 'draw',
      dirty: false,
      zoom: 1,
      panX: 18,
      panY: 18,
      samBoundary: null,
      imageNatural: {w: 0, h: 0}
    };

    const els = {
      taskSelect: document.getElementById('taskSelect'),
      taskHelp: document.getElementById('taskHelp'),
      cameraFilter: document.getElementById('cameraFilter'),
      statusFilter: document.getElementById('statusFilter'),
      search: document.getElementById('search'),
      imageList: document.getElementById('imageList'),
      title: document.getElementById('title'),
      image: document.getElementById('image'),
      overlay: document.getElementById('overlay'),
      wrap: document.getElementById('wrap'),
      stage: document.getElementById('stage'),
      status: document.getElementById('status'),
      saveBtn: document.getElementById('saveBtn'),
      negativeBtn: document.getElementById('negativeBtn'),
      deleteBtn: document.getElementById('deleteBtn'),
      runImagePredBtn: document.getElementById('runImagePredBtn'),
      loadPredBtn: document.getElementById('loadPredBtn'),
      runPipeEndSamBtn: document.getElementById('runPipeEndSamBtn'),
      runSamBoundaryBtn: document.getElementById('runSamBoundaryBtn'),
      clearSamBoundaryBtn: document.getElementById('clearSamBoundaryBtn'),
      trainBtn: document.getElementById('trainBtn'),
      generatePredBtn: document.getElementById('generatePredBtn'),
      badWarpBtn: document.getElementById('badWarpBtn'),
      clearBtn: document.getElementById('clearBtn'),
      prevBtn: document.getElementById('prevBtn'),
      nextBtn: document.getElementById('nextBtn'),
      zoomInBtn: document.getElementById('zoomInBtn'),
      zoomOutBtn: document.getElementById('zoomOutBtn'),
      fitBtn: document.getElementById('fitBtn'),
      modeBtn: document.getElementById('modeBtn'),
      trainStatus: document.getElementById('trainStatus')
    };
    const ctx = els.overlay.getContext('2d');

    function setStatus(text) {
      els.status.innerHTML = state.dirty ? `<span class="dirty">unsaved</span> - ${text}` : text;
    }

    function currentTask() {
      return state.tasks[state.task] || {
        key: state.task,
        class_name: 'pipe_end',
        max_boxes: null,
        camera_filter: null,
        label: state.task,
        allow_negative_labels: false
      };
    }

    function updateTaskHelp() {
      const task = currentTask();
      const maxText = task.max_boxes ? ` Max ${task.max_boxes} box per image.` : '';
      const camText = task.camera_filter ? ` Camera: ${task.camera_filter}.` : '';
      const negText = task.allow_negative_labels ? ' Use Negative image for confirmed no-box frames.' : '';
      els.taskHelp.innerHTML = `Task <b>${task.class_name}</b>. Class is always <b>0</b>.${camText}${maxText}${negText}`;
      if (task.camera_filter) {
        els.cameraFilter.value = task.camera_filter;
        els.cameraFilter.disabled = true;
      } else {
        els.cameraFilter.disabled = false;
      }
      updateNegativeUi();
    }

    function updateNegativeUi() {
      const task = currentTask();
      if (!task.allow_negative_labels) {
        els.negativeBtn.disabled = true;
        els.negativeBtn.textContent = 'Negative image';
        els.negativeBtn.classList.remove('primary');
        return;
      }
      els.negativeBtn.disabled = false;
      els.negativeBtn.textContent = state.negative ? 'Clear negative' : 'Negative image';
      els.negativeBtn.classList.toggle('primary', state.negative);
    }

    function setMode(mode) {
      state.mode = mode === 'pan' ? 'pan' : 'draw';
      if (els.modeBtn) {
        els.modeBtn.textContent = state.mode === 'pan' ? 'Mode: Pan' : 'Mode: Draw';
        els.modeBtn.classList.toggle('primary', state.mode === 'pan');
      }
      els.overlay.style.cursor = state.mode === 'pan' ? 'grab' : 'crosshair';
      setStatus(`${state.mode} mode, zoom ${(state.zoom * 100).toFixed(0)}%, ${state.boxes.length} boxes`);
    }

    async function api(path, options) {
      const res = await fetch(path, options);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      return await res.json();
    }

    function currentItem() {
      return state.filtered[state.currentIndex];
    }

    function applyFilters() {
      const cam = els.cameraFilter.value;
      const status = els.statusFilter.value;
      const q = els.search.value.trim().toLowerCase();
      state.filtered = state.images.filter(item => {
        if (cam !== 'all' && item.camera !== cam) return false;
        if (status === 'todo' && item.labeled) return false;
        if (status === 'done' && !item.labeled) return false;
        if (status === 'negative' && !item.negative) return false;
        if (status === 'bad' && !item.bad_warp) return false;
        if (status === 'usable' && item.bad_warp) return false;
        if (q && !item.rel.toLowerCase().includes(q)) return false;
        return true;
      });
      if (state.currentIndex >= state.filtered.length) state.currentIndex = Math.max(0, state.filtered.length - 1);
      renderList();
    }

    function renderList() {
      els.imageList.innerHTML = '';
      state.filtered.forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = `item ${idx === state.currentIndex ? 'active' : ''} ${item.labeled ? 'done' : ''} ${item.negative ? 'negative' : ''} ${item.bad_warp ? 'bad' : ''}`;
        const badges = [
          `${item.box_count} boxes`,
          item.has_predictions ? `${item.prediction_count} AI` : null,
          item.negative ? 'NEGATIVE' : null,
          item.bad_warp ? 'BAD WARP' : null
        ].filter(Boolean).map(text => `<span class="badge">${text}</span>`).join(' ');
        div.innerHTML = `<div class="item-title">${item.rel}</div>${badges}`;
        div.onclick = () => goTo(idx);
        els.imageList.appendChild(div);
      });
    }

    async function refreshImages(keepRel=null) {
      updateTaskHelp();
      const data = await api('/api/images?task=' + encodeURIComponent(state.task));
      state.images = data.images;
      applyFilters();
      if (keepRel) {
        const idx = state.filtered.findIndex(item => item.rel === keepRel);
        if (idx >= 0) state.currentIndex = idx;
      }
      renderList();
    }

    async function goTo(idx) {
      if (!state.filtered.length) {
        els.title.textContent = 'No images';
        return;
      }
      if (state.dirty) {
        const ok = confirm('You have unsaved boxes. Continue without saving?');
        if (!ok) return;
      }
      state.currentIndex = Math.max(0, Math.min(idx, state.filtered.length - 1));
      state.selected = -1;
      state.drawing = null;
      state.resizing = null;
      state.dirty = false;
      state.negative = false;
      state.samBoundary = null;
      await loadCurrent();
      renderList();
    }

    async function loadCurrent() {
      const item = currentItem();
      if (!item) return;
      els.title.textContent = `${item.bad_warp ? '[BAD WARP] ' : ''}${item.negative ? '[NEGATIVE] ' : ''}${currentTask().class_name} | ${item.rel} (${item.width}x${item.height})`;
      state.imageNatural = {w: item.width, h: item.height};
      els.image.onload = async () => {
        els.image.width = item.width;
        els.image.height = item.height;
        els.overlay.width = item.width;
        els.overlay.height = item.height;
        els.wrap.style.width = `${item.width}px`;
        els.wrap.style.height = `${item.height}px`;
        fitZoom();
        draw();
      };
      els.image.src = item.url + '&t=' + Date.now();
      const labels = await api('/api/labels?task=' + encodeURIComponent(state.task) + '&path=' + encodeURIComponent(item.rel));
      state.boxes = labels.boxes || [];
      state.resizing = null;
      state.negative = !!labels.negative;
      state.samBoundary = null;
      updateNegativeUi();
      setStatus(state.negative ? 'loaded as negative annotation' : `loaded ${state.boxes.length} boxes`);
      draw();
    }

    function fitZoom() {
      const item = currentItem();
      if (!item) return;
      const pad = 46;
      const zw = (els.stage.clientWidth - pad) / item.width;
      const zh = (els.stage.clientHeight - pad) / item.height;
      state.zoom = Math.max(0.1, Math.min(1.6, zw, zh));
      state.panX = Math.round(Math.max(18, (els.stage.clientWidth - item.width * state.zoom) / 2));
      state.panY = Math.round(Math.max(18, (els.stage.clientHeight - item.height * state.zoom) / 2));
      applyZoom();
    }

    function applyZoom() {
      const item = currentItem();
      if (item) {
        els.wrap.style.width = `${item.width}px`;
        els.wrap.style.height = `${item.height}px`;
        els.image.style.width = `${item.width}px`;
        els.image.style.height = `${item.height}px`;
        els.overlay.style.width = `${item.width}px`;
        els.overlay.style.height = `${item.height}px`;
        els.wrap.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
      }
      setStatus(`zoom ${(state.zoom * 100).toFixed(0)}%, ${state.boxes.length} boxes`);
    }

    function canvasPoint(evt) {
      const rect = els.overlay.getBoundingClientRect();
      const scaleX = state.imageNatural.w / Math.max(1, rect.width);
      const scaleY = state.imageNatural.h / Math.max(1, rect.height);
      const x = (evt.clientX - rect.left) * scaleX;
      const y = (evt.clientY - rect.top) * scaleY;
      return {
        x: Math.max(0, Math.min(state.imageNatural.w, x)),
        y: Math.max(0, Math.min(state.imageNatural.h, y))
      };
    }

    function zoomAt(clientX, clientY, factor) {
      const rect = els.stage.getBoundingClientRect();
      const localX = clientX - rect.left;
      const localY = clientY - rect.top;
      const imageX = (localX - state.panX) / Math.max(0.001, state.zoom);
      const imageY = (localY - state.panY) / Math.max(0.001, state.zoom);
      const oldZoom = state.zoom;
      state.zoom = Math.max(0.1, Math.min(6, state.zoom * factor));
      if (Math.abs(state.zoom - oldZoom) < 1e-6) return;
      state.panX = localX - imageX * state.zoom;
      state.panY = localY - imageY * state.zoom;
      applyZoom();
    }

    function normBox(a, b) {
      const x = Math.min(a.x, b.x);
      const y = Math.min(a.y, b.y);
      const w = Math.abs(a.x - b.x);
      const h = Math.abs(a.y - b.y);
      return {class_id: 0, x, y, w, h};
    }

    function hitTest(point) {
      for (let i = state.boxes.length - 1; i >= 0; i--) {
        const b = state.boxes[i];
        if (point.x >= b.x && point.x <= b.x + b.w && point.y >= b.y && point.y <= b.y + b.h) return i;
      }
      return -1;
    }

    function handleRadius() {
      return Math.max(5, 8 / Math.max(0.001, state.zoom));
    }

    function boxHandles(box) {
      const x1 = box.x;
      const y1 = box.y;
      const x2 = box.x + box.w;
      const y2 = box.y + box.h;
      const mx = x1 + box.w / 2;
      const my = y1 + box.h / 2;
      return [
        {id: 'nw', x: x1, y: y1}, {id: 'n', x: mx, y: y1}, {id: 'ne', x: x2, y: y1},
        {id: 'e', x: x2, y: my}, {id: 'se', x: x2, y: y2}, {id: 's', x: mx, y: y2},
        {id: 'sw', x: x1, y: y2}, {id: 'w', x: x1, y: my}
      ];
    }

    function hitHandle(point, box) {
      const r = handleRadius();
      for (const handle of boxHandles(box)) {
        if (Math.abs(point.x - handle.x) <= r && Math.abs(point.y - handle.y) <= r) return handle.id;
      }
      return null;
    }

    function resizeCursor(handle) {
      if (handle === 'n' || handle === 's') return 'ns-resize';
      if (handle === 'e' || handle === 'w') return 'ew-resize';
      if (handle === 'nw' || handle === 'se') return 'nwse-resize';
      if (handle === 'ne' || handle === 'sw') return 'nesw-resize';
      return state.mode === 'pan' ? 'grab' : 'crosshair';
    }

    function updateHoverCursor(evt) {
      if (state.panning || state.drawing || state.resizing || state.mode === 'pan' || state.spaceDown) return;
      const box = state.selected >= 0 ? state.boxes[state.selected] : null;
      if (!box) {
        els.overlay.style.cursor = 'crosshair';
        return;
      }
      const handle = hitHandle(canvasPoint(evt), box);
      els.overlay.style.cursor = handle ? resizeCursor(handle) : 'crosshair';
    }

    function resizedBox(original, handle, point) {
      let x1 = original.x;
      let y1 = original.y;
      let x2 = original.x + original.w;
      let y2 = original.y + original.h;
      if (handle.includes('w')) x1 = point.x;
      if (handle.includes('e')) x2 = point.x;
      if (handle.includes('n')) y1 = point.y;
      if (handle.includes('s')) y2 = point.y;
      x1 = Math.max(0, Math.min(state.imageNatural.w, x1));
      x2 = Math.max(0, Math.min(state.imageNatural.w, x2));
      y1 = Math.max(0, Math.min(state.imageNatural.h, y1));
      y2 = Math.max(0, Math.min(state.imageNatural.h, y2));
      let x = Math.min(x1, x2);
      let y = Math.min(y1, y2);
      let w = Math.abs(x2 - x1);
      let h = Math.abs(y2 - y1);
      if (w < 3) w = 3;
      if (h < 3) h = 3;
      if (x + w > state.imageNatural.w) x = Math.max(0, state.imageNatural.w - w);
      if (y + h > state.imageNatural.h) y = Math.max(0, state.imageNatural.h - h);
      return {...original, x, y, w, h};
    }

    function drawBox(box, idx, selected=false) {
      ctx.save();
      ctx.strokeStyle = selected ? '#32d583' : '#ffd43b';
      ctx.lineWidth = selected ? 3 : 2;
      ctx.fillStyle = selected ? 'rgba(50,213,131,0.12)' : 'rgba(255,212,59,0.08)';
      ctx.fillRect(box.x, box.y, box.w, box.h);
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.fillStyle = selected ? '#32d583' : '#ffd43b';
      ctx.font = '16px Segoe UI';
      ctx.fillText(String(idx + 1), box.x + 3, Math.max(16, box.y - 4));
      if (selected) {
        const r = handleRadius();
        ctx.fillStyle = '#101214';
        ctx.strokeStyle = '#32d583';
        ctx.lineWidth = Math.max(2, 2 / Math.max(0.001, state.zoom));
        for (const handle of boxHandles(box)) {
          ctx.beginPath();
          ctx.rect(handle.x - r, handle.y - r, r * 2, r * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    function drawPolyline(points, color, width) {
      if (!points || points.length < 2) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (const point of points.slice(1)) ctx.lineTo(point[0], point[1]);
      ctx.stroke();
      ctx.restore();
    }

    function drawSamBoundary() {
      const sam = state.samBoundary;
      if (!sam) return;
      ctx.save();
      const guideBoxes = sam.prompt_pipe_end_boxes || [];
      for (const guide of guideBoxes) {
        ctx.strokeStyle = 'rgba(115, 192, 255, 0.95)';
        ctx.lineWidth = 2;
        ctx.strokeRect(guide.x, guide.y, guide.w, guide.h);
      }
      const poly = sam.mask_polygon || [];
      if (poly.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (const point of poly.slice(1)) ctx.lineTo(point[0], point[1]);
        ctx.closePath();
        ctx.fillStyle = 'rgba(50, 213, 131, 0.28)';
        ctx.strokeStyle = 'rgba(50, 213, 131, 0.85)';
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
      }
      if (sam.prompt_box) {
        const box = sam.prompt_box;
        ctx.setLineDash([10, 7]);
        ctx.strokeStyle = 'rgba(255, 99, 71, 0.9)';
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        ctx.setLineDash([]);
        const conf = typeof sam.prompt_conf === 'number' ? ` ${(sam.prompt_conf * 100).toFixed(0)}%` : '';
        const label = sam.prompt_source === 'pipe_end_span' ? `pipe_end span${conf}` : `prompt${conf}`;
        ctx.font = 'bold 18px Segoe UI';
        const metrics = ctx.measureText(label);
        const labelX = Math.max(0, Math.min(box.x, els.overlay.width - metrics.width - 14));
        const labelY = Math.max(22, box.y - 8);
        ctx.fillStyle = 'rgba(255, 99, 71, 0.92)';
        ctx.fillRect(labelX, labelY - 22, metrics.width + 12, 25);
        ctx.fillStyle = '#fff8f3';
        ctx.fillText(label, labelX + 6, labelY - 4);
      }
      ctx.restore();
      drawPolyline(sam.boundary || [], '#ff4fd8', 4);
    }

    function draw() {
      ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
      if (state.negative) {
        ctx.save();
        ctx.fillStyle = 'rgba(115,192,255,0.16)';
        ctx.fillRect(0, 0, els.overlay.width, els.overlay.height);
        ctx.fillStyle = '#d8efff';
        ctx.font = 'bold 28px Segoe UI';
        ctx.fillText('NEGATIVE ANNOTATION', 18, 42);
          ctx.restore();
      }
      drawSamBoundary();
      state.boxes.forEach((box, idx) => drawBox(box, idx, idx === state.selected));
      if (state.drawing) {
        const box = normBox(state.drawing.start, state.drawing.end);
        drawBox(box, state.boxes.length, true);
      }
    }

    function markDirty() {
      state.dirty = true;
      draw();
      setStatus(`${state.boxes.length} boxes`);
    }

    async function save() {
      const item = currentItem();
      if (!item) return;
      const result = await api('/api/labels', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          path: item.rel,
          task: state.task,
          width: state.imageNatural.w,
          height: state.imageNatural.h,
          boxes: state.boxes,
          negative: state.negative && state.boxes.length === 0
        })
      });
      state.dirty = false;
      state.negative = !!result.negative;
      updateNegativeUi();
      await refreshImages(item.rel);
      setStatus(state.negative ? 'saved negative annotation' : `saved ${state.boxes.length} boxes`);
      await refreshTrainStatus();
    }

    async function toggleNegativeAnnotation() {
      const item = currentItem();
      if (!item) return;
      const task = currentTask();
      if (!task.allow_negative_labels) {
        setStatus('negative annotations are not enabled for this task');
        return;
      }
      const next = !state.negative;
      if (next && (state.boxes.length || state.dirty)) {
        const ok = confirm('Mark this image as negative and clear current boxes?');
        if (!ok) return;
      }
      const result = await api('/api/labels', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          path: item.rel,
          task: state.task,
          width: state.imageNatural.w,
          height: state.imageNatural.h,
          boxes: [],
          negative: next
        })
      });
      state.boxes = [];
      state.selected = -1;
      state.drawing = null;
      state.resizing = null;
      state.dirty = false;
      state.negative = !!result.negative;
      updateNegativeUi();
      draw();
      await refreshImages(item.rel);
      await loadCurrent();
      setStatus(state.negative ? 'marked as negative annotation' : 'negative annotation cleared');
      await refreshTrainStatus();
    }

    async function refreshTrainStatus() {
      try {
        const train = await api('/api/train-status');
        const prefix = train.running ? 'AI training' : train.status === 'ready' ? 'AI ready' : `AI ${train.status}`;
        els.trainStatus.textContent = `${prefix}: ${train.message}`;
        if (els.trainBtn) {
          els.trainBtn.disabled = !!train.running;
          els.trainBtn.textContent = train.running ? 'AI busy' : 'Train AI';
          els.trainBtn.classList.toggle('primary', train.status === 'training');
        }
        if (els.generatePredBtn) {
          els.generatePredBtn.disabled = !!train.running;
          els.generatePredBtn.textContent = train.running ? 'AI busy' : 'Generate predictions';
          els.generatePredBtn.classList.toggle('primary', train.status === 'predicting');
        }
        return train;
      } catch (err) {
        els.trainStatus.textContent = `AI status error: ${err.message}`;
        return null;
      }
    }

    async function trainNow() {
      try {
        setStatus('requesting AI update...');
        const result = await api('/api/train', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({reason: 'manual button', task: state.task})
        });
        setStatus(result.scheduled ? 'AI training started' : `AI training not started: ${result.reason || 'unknown'}`);
        await refreshTrainStatus();
      } catch (err) {
        setStatus(`AI train request failed: ${err.message}`);
      }
    }

    async function generatePredictionsNow() {
      try {
        setStatus('requesting AI predictions...');
        const result = await api('/api/generate-predictions', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({reason: 'manual button', task: state.task})
        });
        setStatus(result.scheduled ? 'AI prediction generation started' : `AI predictions not started: ${result.reason || 'unknown'}`);
        await refreshTrainStatus();
      } catch (err) {
        setStatus(`AI prediction request failed: ${err.message}`);
      }
    }

    async function runModelOnCurrentImage() {
      const item = currentItem();
      if (!item) return;
      if (state.dirty || state.boxes.length) {
        const ok = confirm('Replace current boxes with model predictions for this image? Unsaved changes will be lost.');
        if (!ok) return;
      }
      const button = els.runImagePredBtn;
      try {
        if (button) {
          button.disabled = true;
          button.textContent = 'Running model...';
        }
        setStatus('running model on current image...');
        const payload = await api('/api/predict-current', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({path: item.rel, task: state.task})
        });
        state.boxes = payload.boxes || [];
        state.selected = -1;
        state.resizing = null;
        state.negative = false;
        updateNegativeUi();
        state.dirty = true;
        draw();
        await refreshImages(item.rel);
        const post = payload.postprocess || {};
        const extras = [
          post.isolated_filter?.removed_count ? `outlier -${post.isolated_filter.removed_count}` : null,
          post.gap_recovery?.recovered_count ? `gap +${post.gap_recovery.recovered_count}` : null,
          post.edge_gap_recovery?.recovered_count ? `edge +${post.edge_gap_recovery.recovered_count}` : null,
          post.large_box_split?.added_count ? `split +${post.large_box_split.added_count}` : null
        ].filter(Boolean).join(', ');
        setStatus(`model found ${state.boxes.length} boxes${extras ? ` (${extras})` : ''}; review and save`);
      } catch (err) {
        setStatus(`model prediction failed: ${err.message}`);
      } finally {
        if (button) {
          button.disabled = false;
          button.textContent = 'Run model on this image';
        }
      }
    }

    function modelPostprocessExtras(post) {
      return [
        post.isolated_filter?.removed_count ? `outlier -${post.isolated_filter.removed_count}` : null,
        post.gap_recovery?.recovered_count ? `gap +${post.gap_recovery.recovered_count}` : null,
        post.edge_gap_recovery?.recovered_count ? `edge +${post.edge_gap_recovery.recovered_count}` : null,
        post.large_box_split?.added_count ? `split +${post.large_box_split.added_count}` : null
      ].filter(Boolean).join(', ');
    }

    async function applyModelPrediction(payload, item) {
      state.boxes = payload.boxes || [];
      state.selected = -1;
      state.resizing = null;
      state.negative = false;
      updateNegativeUi();
      state.dirty = true;
      draw();
      await refreshImages(item.rel);
      return modelPostprocessExtras(payload.postprocess || {});
    }

    async function runSamBoundaryOnCurrentImage(promptMode='pipe_end_span') {
      const item = currentItem();
      if (!item) return;
      const usePipeEndPrompt = promptMode !== 'full_image';
      const button = usePipeEndPrompt ? els.runPipeEndSamBtn : els.runSamBoundaryBtn;
      try {
        if (button) {
          button.disabled = true;
          button.textContent = usePipeEndPrompt ? 'Running pipe-end...' : 'Running SAM...';
        }
        setStatus(usePipeEndPrompt ? 'running pipe-end prompt + SAM boundary...' : 'running full-image SAM boundary...');
        const payload = await api('/api/sam-boundary', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({path: item.rel, side: 'right', prompt_mode: promptMode})
        });
        state.samBoundary = payload;
        draw();
        const source = payload.prompt_source === 'pipe_end_span' ? `pipe-end span, ${payload.prompt_pipe_end_count || 0} pipe ends`
          : `full image, ${payload.mask_count || 0} masks`;
        setStatus(`SAM boundary ready (${source}, ${payload.boundary?.length || 0} points); running guided model...`);
        if (state.dirty && !confirm('Replace unsaved boxes with SAM-guided model predictions?')) {
          setStatus(`SAM boundary ready (${source}, ${payload.boundary?.length || 0} points); boxes unchanged`);
          return;
        }
        const prediction = await api('/api/predict-current', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({path: item.rel, task: state.task})
        });
        const extras = await applyModelPrediction(prediction, item);
        setStatus(`SAM-guided model found ${state.boxes.length} boxes${extras ? ` (${extras})` : ''}; review and save`);
      } catch (err) {
        setStatus(`SAM boundary failed: ${err.message}`);
      } finally {
        if (button) {
          button.disabled = false;
          button.textContent = usePipeEndPrompt ? 'Pipe-end SAM' : 'Full-image SAM';
        }
      }
    }

    async function setBadWarp(flag) {
      const item = currentItem();
      if (!item) return;
      await api('/api/status', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path: item.rel, bad_warp: !!flag})
      });
      await refreshImages(item.rel);
      await loadCurrent();
      setStatus(flag ? 'marked as bad warp' : 'bad warp cleared');
    }

    async function loadPredictions() {
      const item = currentItem();
      if (!item) return;
      if (state.dirty) {
        const ok = confirm('Current boxes are unsaved. Replace them with AI predictions?');
        if (!ok) return;
      }
      const payload = await api('/api/predictions?task=' + encodeURIComponent(state.task) + '&path=' + encodeURIComponent(item.rel));
      state.boxes = payload.boxes || [];
      state.selected = -1;
      state.resizing = null;
      state.negative = false;
      updateNegativeUi();
      markDirty();
      setStatus(`loaded ${state.boxes.length} AI boxes; review and save`);
    }

    function deleteSelected() {
      if (state.selected < 0) return;
      state.boxes.splice(state.selected, 1);
      state.selected = -1;
      state.resizing = null;
      markDirty();
    }

    function beginPan(evt) {
      evt.preventDefault();
      state.panning = {
        x: evt.clientX,
        y: evt.clientY,
        panX: state.panX,
        panY: state.panY
      };
      els.overlay.style.cursor = 'grabbing';
      els.stage.style.cursor = 'grabbing';
    }

    function shouldPan(evt) {
      return evt.button === 1 || evt.button === 2 || (evt.button === 0 && (state.spaceDown || state.mode === 'pan' || evt.shiftKey || evt.altKey));
    }

    els.stage.addEventListener('mousedown', evt => {
      if (shouldPan(evt)) {
        beginPan(evt);
        return;
      }
    });

    els.overlay.addEventListener('mousedown', evt => {
      if (shouldPan(evt)) {
        evt.preventDefault();
        beginPan(evt);
        return;
      }
      if (evt.button !== 0) return;
      const p = canvasPoint(evt);
      if (state.selected >= 0 && state.boxes[state.selected]) {
        const handle = hitHandle(p, state.boxes[state.selected]);
        if (handle) {
          evt.preventDefault();
          state.resizing = {
            index: state.selected,
            handle,
            original: {...state.boxes[state.selected]}
          };
          els.overlay.style.cursor = resizeCursor(handle);
          return;
        }
      }
      const hit = hitTest(p);
      if (hit >= 0) {
        state.selected = hit;
        draw();
        return;
      }
      state.selected = -1;
      state.drawing = {start: p, end: p};
      draw();
    });

    window.addEventListener('mousemove', evt => {
      if (state.panning) {
        evt.preventDefault();
        state.panX = state.panning.panX + (evt.clientX - state.panning.x);
        state.panY = state.panning.panY + (evt.clientY - state.panning.y);
        applyZoom();
        return;
      }
      if (state.resizing) {
        evt.preventDefault();
        state.boxes[state.resizing.index] = resizedBox(state.resizing.original, state.resizing.handle, canvasPoint(evt));
        draw();
        return;
      }
      updateHoverCursor(evt);
      if (!state.drawing) return;
      state.drawing.end = canvasPoint(evt);
      draw();
    });

    window.addEventListener('mouseup', () => {
      if (state.panning) {
        state.panning = null;
        els.overlay.style.cursor = state.mode === 'pan' ? 'grab' : 'crosshair';
        els.stage.style.cursor = 'default';
        return;
      }
      if (state.resizing) {
        state.resizing = null;
        markDirty();
        els.overlay.style.cursor = state.mode === 'pan' ? 'grab' : 'crosshair';
        return;
      }
      if (!state.drawing) return;
      const box = normBox(state.drawing.start, state.drawing.end);
      state.drawing = null;
      if (box.w >= 3 && box.h >= 3) {
        const task = currentTask();
        if (task.max_boxes === 1) {
          state.boxes = [box];
          state.selected = 0;
        } else {
          state.boxes.push(box);
          state.selected = state.boxes.length - 1;
        }
        state.negative = false;
        updateNegativeUi();
        markDirty();
      } else {
        draw();
      }
    });

    els.overlay.addEventListener('auxclick', evt => {
      if (evt.button === 1) evt.preventDefault();
    });

    els.stage.addEventListener('contextmenu', evt => {
      if (state.panning || evt.target === els.overlay || evt.target === els.image) evt.preventDefault();
    });

    els.stage.addEventListener('wheel', evt => {
      evt.preventDefault();
      const factor = evt.deltaY < 0 ? 1.12 : 1 / 1.12;
      zoomAt(evt.clientX, evt.clientY, factor);
    }, { passive: false });

    els.saveBtn.onclick = save;
    els.negativeBtn.onclick = toggleNegativeAnnotation;
    els.deleteBtn.onclick = deleteSelected;
    els.runImagePredBtn.onclick = runModelOnCurrentImage;
    els.loadPredBtn.onclick = loadPredictions;
    els.runPipeEndSamBtn.onclick = () => runSamBoundaryOnCurrentImage('pipe_end_span');
    els.runSamBoundaryBtn.onclick = () => runSamBoundaryOnCurrentImage('full_image');
    els.clearSamBoundaryBtn.onclick = () => {
      state.samBoundary = null;
      draw();
      setStatus('SAM boundary cleared');
    };
    els.trainBtn.onclick = trainNow;
    els.generatePredBtn.onclick = generatePredictionsNow;
    els.badWarpBtn.onclick = async () => {
      const item = currentItem();
      if (!item) return;
      await setBadWarp(!item.bad_warp);
    };
    els.clearBtn.onclick = () => {
      if (state.boxes.length && confirm('Clear all boxes for this image?')) {
        state.boxes = [];
        state.selected = -1;
        state.resizing = null;
        markDirty();
      }
    };
    els.prevBtn.onclick = () => goTo(state.currentIndex - 1);
    els.nextBtn.onclick = () => goTo(state.currentIndex + 1);
    els.zoomInBtn.onclick = () => {
      const rect = els.stage.getBoundingClientRect();
      zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1.2);
    };
    els.zoomOutBtn.onclick = () => {
      const rect = els.stage.getBoundingClientRect();
      zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1 / 1.2);
    };
    els.fitBtn.onclick = fitZoom;
    els.modeBtn.onclick = () => setMode(state.mode === 'pan' ? 'draw' : 'pan');

    els.taskSelect.addEventListener('change', async () => {
      if (state.dirty) {
        const ok = confirm('You have unsaved boxes. Continue without saving?');
        if (!ok) {
          els.taskSelect.value = state.task;
          return;
        }
      }
      state.task = els.taskSelect.value;
      state.currentIndex = 0;
      state.selected = -1;
      state.drawing = null;
      state.resizing = null;
      state.dirty = false;
      state.negative = false;
      state.samBoundary = null;
      await refreshImages();
      await loadCurrent();
      renderList();
      await refreshTrainStatus();
    });

    for (const el of [els.cameraFilter, els.statusFilter, els.search]) {
      el.addEventListener('input', async () => {
        const rel = currentItem()?.rel;
        applyFilters();
        if (rel) {
          const idx = state.filtered.findIndex(item => item.rel === rel);
          state.currentIndex = idx >= 0 ? idx : 0;
        }
        await loadCurrent();
        renderList();
      });
    }

    window.addEventListener('keydown', evt => {
      const tag = document.activeElement?.tagName?.toLowerCase();
      const typing = tag === 'input' || tag === 'select' || tag === 'textarea';
      if (evt.code === 'Space' && !typing) {
        evt.preventDefault();
        state.spaceDown = true;
        els.overlay.style.cursor = 'grab';
        return;
      } else if (evt.key.toLowerCase() === 'p' && !typing) {
        evt.preventDefault();
        setMode(state.mode === 'pan' ? 'draw' : 'pan');
        return;
      }
      if (evt.ctrlKey && evt.key.toLowerCase() === 's') {
        evt.preventDefault();
        save();
      } else if ((evt.key === 'Delete' || evt.key === 'Backspace' || evt.key === 'Del' || evt.code === 'Delete') && !typing) {
        evt.preventDefault();
        deleteSelected();
      } else if (evt.key === 'ArrowRight' && !typing) {
        goTo(state.currentIndex + 1);
      } else if (evt.key === 'ArrowLeft' && !typing) {
        goTo(state.currentIndex - 1);
      } else if (evt.key === 'Escape' && !typing) {
        state.drawing = null;
        state.resizing = null;
        state.selected = -1;
        draw();
      }
    });

    window.addEventListener('keyup', evt => {
      if (evt.code === 'Space') {
        state.spaceDown = false;
        if (!state.panning) els.overlay.style.cursor = state.mode === 'pan' ? 'grab' : 'crosshair';
      }
    });

    async function boot() {
      try {
        const taskData = await api('/api/tasks');
        state.tasks = {};
        for (const task of taskData.tasks || []) {
          state.tasks[task.key] = task;
        }
        updateTaskHelp();
        await refreshImages();
        await loadCurrent();
        await refreshTrainStatus();
        setInterval(refreshTrainStatus, 5000);
      } catch (err) {
        els.title.textContent = 'Error';
        setStatus(err.message);
        console.error(err);
      }
    }
    setMode('draw');
    boot();
  </script>
</body>
</html>
"""


class AnnotatorHandler(BaseHTTPRequestHandler):
    paths: AppPaths

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/plain; charset=utf-8") -> None:
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                self.send_text(HTML, content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/tasks":
                self.send_json({"tasks": [task.to_json() for task in list_tasks(self.paths)]})
                return
            if parsed.path == "/api/images":
                query = parse_qs(parsed.query)
                task = task_config(self.paths, query.get("task", ["pipe_end"])[0])
                self.send_json({"images": build_image_items(self.paths, task), "task": task.to_json()})
                return
            if parsed.path == "/api/labels":
                query = parse_qs(parsed.query)
                task = task_config(self.paths, query.get("task", ["pipe_end"])[0])
                rel = normalize_rel(query.get("path", [""])[0])
                image_path = self.paths.images_root / rel
                if not image_path.exists():
                    self.send_json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
                    return
                width, height = image_size(image_path)
                label_path = (task.labels_root / rel).with_suffix(".txt")
                status = load_status(self.paths.status_path)
                self.send_json(
                    {
                        "boxes": yolo_to_boxes(label_path, width, height),
                        "width": width,
                        "height": height,
                        "negative": is_negative_annotation(status, task, rel),
                    }
                )
                return
            if parsed.path == "/api/predictions":
                query = parse_qs(parsed.query)
                task = task_config(self.paths, query.get("task", ["pipe_end"])[0])
                rel = normalize_rel(query.get("path", [""])[0])
                image_path = self.paths.images_root / rel
                if not image_path.exists():
                    self.send_json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
                    return
                width, height = image_size(image_path)
                prediction_path = (task.predictions_root / rel).with_suffix(".txt")
                self.send_json(
                    {
                        "boxes": yolo_to_boxes(prediction_path, width, height),
                        "width": width,
                        "height": height,
                        "prediction_path": repo_display_path(prediction_path) if prediction_path.exists() else None,
                    }
                )
                return
            if parsed.path == "/api/status":
                query = parse_qs(parsed.query)
                rel = normalize_rel(query.get("path", [""])[0])
                status = load_status(self.paths.status_path)
                self.send_json({"bad_warp": bool(status.get("bad_warp", {}).get(rel.as_posix(), False))})
                return
            if parsed.path == "/api/train-status":
                self.send_json(TRAINING_STATE.snapshot())
                return
            if parsed.path == "/image":
                query = parse_qs(parsed.query)
                rel = normalize_rel(query.get("path", [""])[0])
                image_path = self.paths.images_root / rel
                if not image_path.exists() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    self.send_json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
                    return
                ctype = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                data = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/roi-picker":
                self.send_text(ROI_PICKER_HTML, content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/roi-info":
                query = parse_qs(parsed.query)
                camera = query.get("camera", ["cam151"])[0]
                toml_path = _roi_toml_for_camera(self.paths, camera)
                if toml_path is None or not toml_path.exists():
                    self.send_json({"error": "TOML no disponible. Usa --roi-toml-151 / --roi-toml-152."})
                    return
                try:
                    roi = _read_roi_toml(toml_path)
                    dst_rect = roi.get("dst_rect_override")
                    src_pts = roi.get("src_points_override")
                    raw_path = _raw_image_for_camera(self.paths, camera)
                    if raw_path and raw_path.exists() and src_pts:
                        _, view_rect, (w, h) = _build_full_warp_jpeg(raw_path, src_pts)
                        source_type = "raw"
                    else:
                        cam_dir = self.paths.images_root / camera
                        imgs = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.jpeg"))
                        if not imgs:
                            self.send_json({"error": "Sin imágenes de anotación para esta cámara."})
                            return
                        w, h = image_size(imgs[-1])
                        view_rect = list(dst_rect) if dst_rect else [0.0, 0.0, float(w), float(h)]
                        source_type = "annotation"
                    self.send_json({"dst_rect": dst_rect, "view_rect": view_rect,
                                    "image_w": w, "image_h": h, "source_type": source_type})
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if parsed.path == "/roi-warp-image":
                query = parse_qs(parsed.query)
                camera = query.get("camera", ["cam151"])[0]
                try:
                    toml_path = _roi_toml_for_camera(self.paths, camera)
                    raw_path = _raw_image_for_camera(self.paths, camera)
                    if toml_path and toml_path.exists() and raw_path and raw_path.exists():
                        roi = _read_roi_toml(toml_path)
                        src_pts = roi.get("src_points_override")
                        if src_pts:
                            jpeg_bytes, _, _ = _build_full_warp_jpeg(raw_path, src_pts)
                            self.send_response(HTTPStatus.OK)
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(jpeg_bytes)))
                            self.end_headers()
                            self.wfile.write(jpeg_bytes)
                            return
                    cam_dir = self.paths.images_root / camera
                    imgs = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.jpeg"))
                    if not imgs:
                        self.send_json({"error": "Sin imágenes"}, HTTPStatus.NOT_FOUND)
                        return
                    data = imgs[-1].read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if parsed.path == "/api/status":
                rel = normalize_rel(str(payload.get("path", "")))
                image_path = self.paths.images_root / rel
                if not image_path.exists():
                    self.send_json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
                    return
                status = load_status(self.paths.status_path)
                bad_warp = status.setdefault("bad_warp", {})
                rel_key = rel.as_posix()
                if bool(payload.get("bad_warp", False)):
                    bad_warp[rel_key] = True
                else:
                    bad_warp.pop(rel_key, None)
                save_status(self.paths.status_path, status)
                self.send_json({"ok": True, "path": rel_key, "bad_warp": bool(bad_warp.get(rel_key, False))})
                return
            if parsed.path == "/api/train":
                reason = str(payload.get("reason", "manual request"))
                self.send_json(schedule_training(self.paths, reason, str(payload.get("task", "pipe_end"))))
                return
            if parsed.path == "/api/generate-predictions":
                reason = str(payload.get("reason", "manual request"))
                self.send_json(schedule_prediction(self.paths, reason, str(payload.get("task", "pipe_end"))))
                return
            if parsed.path == "/api/predict-current":
                rel = normalize_rel(str(payload.get("path", "")))
                self.send_json(run_single_image_prediction(self.paths, rel, str(payload.get("task", "pipe_end"))))
                return
            if parsed.path == "/api/sam-boundary":
                rel = normalize_rel(str(payload.get("path", "")))
                side = str(payload.get("side", "right"))
                sam_model = str(payload.get("sam_model", "sam2.1_s.pt"))
                prompt_mode = str(payload.get("prompt_mode", "pipe_end_span"))
                self.send_json(run_sam_boundary(self.paths, rel, side=side, sam_model=sam_model, prompt_mode=prompt_mode))
                return
            if parsed.path == "/api/roi-save":
                camera = str(payload.get("camera", "cam151"))
                dst_rect = payload.get("dst_rect", [])
                toml_path = _roi_toml_for_camera(self.paths, camera)
                if toml_path is None:
                    self.send_json({"error": "TOML no configurado para esta cámara."}, HTTPStatus.BAD_REQUEST)
                    return
                if len(dst_rect) != 4:
                    self.send_json({"error": "dst_rect debe tener 4 valores."}, HTTPStatus.BAD_REQUEST)
                    return
                _save_roi_dst_rect(toml_path, [float(v) for v in dst_rect])
                self.send_json({"ok": True, "dst_rect": dst_rect, "toml": str(toml_path)})
                return
            if parsed.path != "/api/labels":
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            task = task_config(self.paths, str(payload.get("task", "pipe_end")))
            rel = normalize_rel(str(payload.get("path", "")))
            image_path = self.paths.images_root / rel
            if not image_path.exists():
                self.send_json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
                return
            if task.camera_filter and (not rel.parts or rel.parts[0] != task.camera_filter):
                self.send_json({"error": f"{task.label} only supports {task.camera_filter} images."}, HTTPStatus.BAD_REQUEST)
                return
            width, height = image_size(image_path)
            boxes = payload.get("boxes", [])
            if not isinstance(boxes, list):
                raise ValueError("boxes must be a list")
            negative = bool(payload.get("negative", False))
            if negative and not task.allow_negative_labels:
                raise ValueError(f"{task.label} does not support explicit negative annotations")
            if negative:
                boxes = []
            if task.max_boxes is not None and len(boxes) > task.max_boxes:
                raise ValueError(f"{task.label} supports at most {task.max_boxes} box per image")
            label_path = (task.labels_root / rel).with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(boxes_to_yolo(boxes, width, height), encoding="utf-8")
            status = load_status(self.paths.status_path)
            negative = set_negative_annotation(status, task, rel, negative)
            save_status(self.paths.status_path, status)
            self.send_json(
                {
                    "ok": True,
                    "label_path": repo_display_path(label_path),
                    "count": len(boxes),
                    "negative": negative,
                    "task": task.to_json(),
                    "training": {"scheduled": False, "reason": "manual training only"},
                }
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def find_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local YOLO box annotation app for pipe_end.")
    parser.add_argument("--root", type=Path, default=PIPE_END_ROOT)
    parser.add_argument("--images-root", type=Path, default=Path("annotation_pool/images"))
    parser.add_argument("--labels-root", type=Path, default=Path("annotation_pool/labels"))
    parser.add_argument("--predictions-root", type=Path, default=Path("predictions/current/labels"))
    parser.add_argument("--status-path", type=Path, default=Path("annotation_pool/image_status.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--min-train-images", type=int, default=4)
    parser.add_argument("--train-epochs", type=int, default=40)
    parser.add_argument("--train-imgsz", type=int, default=1280)
    parser.add_argument("--train-batch", type=int, default=1)
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--train-device", default="0", help="CUDA device index or 'cpu'. Default 0 (first GPU).")
    parser.add_argument("--roi-toml-151", type=Path, default=None, help="Path to cam151 ROI TOML file.")
    parser.add_argument("--roi-toml-152", type=Path, default=None, help="Path to cam152 ROI TOML file.")
    parser.add_argument("--raw-image-151", type=Path, default=None, help="Raw camera image for cam151 full-warp preview.")
    parser.add_argument("--raw-image-152", type=Path, default=None, help="Raw camera image for cam152 full-warp preview.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    images_root = (root / args.images_root).resolve() if not args.images_root.is_absolute() else args.images_root.resolve()
    labels_root = (root / args.labels_root).resolve() if not args.labels_root.is_absolute() else args.labels_root.resolve()
    predictions_root = (
        (root / args.predictions_root).resolve()
        if not args.predictions_root.is_absolute()
        else args.predictions_root.resolve()
    )
    status_path = (root / args.status_path).resolve() if not args.status_path.is_absolute() else args.status_path.resolve()
    if not images_root.exists():
        raise FileNotFoundError(images_root)
    labels_root.mkdir(parents=True, exist_ok=True)
    predictions_root.mkdir(parents=True, exist_ok=True)

    port = find_free_port(args.port)
    AnnotatorHandler.paths = AppPaths(
        root=root,
        images_root=images_root,
        labels_root=labels_root,
        predictions_root=predictions_root,
        status_path=status_path,
        auto_train=False,
        min_train_images=args.min_train_images,
        train_epochs=args.train_epochs,
        train_imgsz=args.train_imgsz,
        train_batch=args.train_batch,
        base_model=args.base_model,
        train_device=args.train_device,
        roi_toml_151=args.roi_toml_151.resolve() if args.roi_toml_151 else None,
        roi_toml_152=args.roi_toml_152.resolve() if args.roi_toml_152 else None,
        raw_image_151=args.raw_image_151.resolve() if args.raw_image_151 else None,
        raw_image_152=args.raw_image_152.resolve() if args.raw_image_152 else None,
    )
    for task in list_tasks(AnnotatorHandler.paths):
        task.labels_root.mkdir(parents=True, exist_ok=True)
        task.predictions_root.mkdir(parents=True, exist_ok=True)
        task.work_root.mkdir(parents=True, exist_ok=True)
        task.project_root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, port), AnnotatorHandler)
    url = f"http://{args.host}:{port}/"
    print(f"Pipe End Annotator running at {url}")
    print(f"images: {images_root}")
    print(f"labels: {labels_root}")
    print(f"predictions: {predictions_root}")
    print(f"status: {status_path}")
    print(f"auto_train: False, min_images={args.min_train_images}, epochs={args.train_epochs}")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
