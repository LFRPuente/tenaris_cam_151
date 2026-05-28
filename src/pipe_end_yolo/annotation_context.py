from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SPACING_STATS_CACHE: dict[tuple[str, str, str, int, int], dict[str, Any] | None] = {}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pipe_end_root() -> Path:
    return repo_root() / "pipe_end_detection"


def normalize_pipe_end_side(side: str | int | None) -> str:
    raw = str(side or "").strip().lower()
    if raw in {"151", "cam151", "cam_151", "left"}:
        return "151"
    if raw in {"152", "cam152", "cam_152", "right"}:
        return "152"
    raise ValueError(f"Unsupported pipe-end camera side: {side!r}")


def camera_filter_for_side(side: str | int | None) -> str:
    return f"cam{normalize_pipe_end_side(side)}"


def task_key_for_side(side: str | int | None) -> str:
    return "pipe_end_cam152" if normalize_pipe_end_side(side) == "152" else "pipe_end"


def model_path_for_side(side: str | int | None) -> Path:
    side_key = normalize_pipe_end_side(side)
    camera_env = os.environ.get(f"PIPE_END_YOLO_MODEL_CAM{side_key}")
    if camera_env:
        candidate = Path(camera_env)
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"No existe PIPE_END_YOLO_MODEL_CAM{side_key}: {camera_env}")

    if side_key == "152":
        candidate = repo_root() / "models" / "pipe_end_cam152_active" / "best.pt"
    else:
        candidate = repo_root() / "models" / "pipe_end_active" / "best.pt"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"No existe modelo pipe_end para cam{side_key}: {candidate}")


def annotation_images_root() -> Path:
    return pipe_end_root() / "annotation_pool" / "images"


def annotation_labels_root() -> Path:
    return pipe_end_root() / "annotation_pool" / "labels"


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    return int(width), int(height)


def _image_path_for_label_stem(images_root: Path, rel_stem: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = (images_root / rel_stem).with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _median(values: list[float]) -> float | None:
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


def _yolo_label_center_y(line: str, image_height: int) -> float | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        y_center_norm = float(parts[2])
    except ValueError:
        return None
    return float(y_center_norm) * float(image_height)


def spacing_stats_from_annotations(
    *,
    images_root: Path,
    labels_root: Path,
    camera_filter: str | None = None,
) -> dict[str, Any] | None:
    labels_root = Path(labels_root)
    images_root = Path(images_root)
    if not labels_root.exists() or not images_root.exists():
        return None

    label_files: list[Path] = []
    for path in labels_root.rglob("*.txt"):
        if not path.is_file():
            continue
        rel = path.relative_to(labels_root).with_suffix("")
        if camera_filter and (not rel.parts or rel.parts[0] != camera_filter):
            continue
        label_files.append(path)
    if not label_files:
        return None

    latest_mtime = max(path.stat().st_mtime_ns for path in label_files)
    cache_key = (
        str(images_root.resolve()),
        str(labels_root.resolve()),
        str(camera_filter or ""),
        len(label_files),
        int(latest_mtime),
    )
    if cache_key in _SPACING_STATS_CACHE:
        return _SPACING_STATS_CACHE[cache_key]

    spacings: list[float] = []
    sample_count = 0
    for label_path in label_files:
        rel_stem = label_path.relative_to(labels_root).with_suffix("")
        image_path = _image_path_for_label_stem(images_root, rel_stem)
        if image_path is None:
            continue
        try:
            _width, height = _image_size(image_path)
        except Exception:
            continue
        ys: list[float] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            y = _yolo_label_center_y(line, height)
            if y is not None:
                ys.append(float(y))
        if len(ys) < 2:
            continue
        ys.sort()
        gaps = [ys[idx + 1] - ys[idx] for idx in range(len(ys) - 1) if ys[idx + 1] > ys[idx]]
        if gaps:
            sample_count += 1
            spacings.extend(gaps)

    raw_median = _median(spacings)
    if raw_median is None or len(spacings) < 10:
        _SPACING_STATS_CACHE[cache_key] = None
        return None

    central = [gap for gap in spacings if 0.35 * raw_median <= gap <= 2.50 * raw_median]
    median_gap = _median(central) or raw_median
    mean_gap, variance_gap = _mean_variance(central)
    deviations = [abs(gap - median_gap) for gap in central]
    mad = _median(deviations) or 0.0
    robust_sigma = 1.4826 * float(mad)
    std_gap = float(variance_gap) ** 0.5
    allowed_gap = max(1.65 * float(median_gap), float(median_gap) + 4.0 * robust_sigma)
    if std_gap > 0:
        allowed_gap = max(allowed_gap, float(mean_gap) + 2.0 * std_gap)
    allowed_gap = min(allowed_gap, 3.0 * float(median_gap))

    stats: dict[str, Any] = {
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
    _SPACING_STATS_CACHE[cache_key] = stats
    return stats


def spacing_stats_for_side(side: str | int | None) -> dict[str, Any] | None:
    return spacing_stats_from_annotations(
        images_root=annotation_images_root(),
        labels_root=annotation_labels_root(),
        camera_filter=camera_filter_for_side(side),
    )
