from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PIPE_END_CLASS_ID = 0
PIPE_END_CLASS_NAME = "pipe_end"
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.50
DEFAULT_OVERLAP_SUPPRESSION = 0.45
DEFAULT_CONTAINMENT_RATIO = 0.85
DEFAULT_CONTAINED_CHILDREN = 2
DEFAULT_VERTICAL_DUPLICATE_HEIGHT_RATIO = 0.50
DEFAULT_VERTICAL_DUPLICATE_OVERLAP = 0.30
DEFAULT_VERTICAL_DUPLICATE_Y_OVERLAP = 0.65
DEFAULT_GAP_RECOVERY_ENABLED = True
DEFAULT_GAP_RECOVERY_CONF = 0.05
DEFAULT_GAP_RECOVERY_MIN_RATIO = 1.35
DEFAULT_GAP_RECOVERY_MAX_RATIO = 4.25
DEFAULT_GAP_RECOVERY_EMPTY_GAP_RATIO = 0.70
DEFAULT_GAP_RECOVERY_BAND_RATIO = 0.70
DEFAULT_GAP_RECOVERY_MAX_MISSING_PER_GAP = 2
DEFAULT_GAP_RECOVERY_PITCH_FALLBACK_ENABLED = True
DEFAULT_GAP_RECOVERY_PITCH_FALLBACK_CONF = 0.025
DEFAULT_ISOLATED_FILTER_ENABLED = True
DEFAULT_ISOLATED_FILTER_MIN_CLUSTER_SIZE = 3
DEFAULT_ISOLATED_FILTER_GAP_RATIO = 2.8
DEFAULT_ISOLATED_FILTER_MIN_GAP_PX = 55.0
DEFAULT_FAR_X_CONF_FILTER_ENABLED = True
DEFAULT_FAR_X_CONF_FILTER_DISTANCE_RATIO = 1.8
DEFAULT_FAR_X_CONF_FILTER_MIN_DISTANCE_PX = 35.0
DEFAULT_FAR_X_CONF_FILTER_MIN_CONF = 0.25
DEFAULT_FAR_X_CONF_FILTER_EXTRA_CONF = 0.12
DEFAULT_EDGE_GAP_RECOVERY_ENABLED = True
DEFAULT_EDGE_GAP_RECOVERY_MIN_RATIO = 1.15
DEFAULT_EDGE_GAP_RECOVERY_EDGE_SPACE_RATIO = 0.65
DEFAULT_EDGE_GAP_RECOVERY_MAX_MISSING_PER_EDGE = 3
DEFAULT_LARGE_BOX_SPLIT_ENABLED = True
DEFAULT_LARGE_BOX_SPLIT_HEIGHT_RATIO = 1.65
DEFAULT_LARGE_BOX_SPLIT_MAX_PARTS = 4
DEFAULT_LARGE_BOX_SPLIT_MIN_EDGE_STRENGTH_RATIO = 0.55
DEFAULT_LARGE_BOX_SPLIT_CLOSE_X_RATIO = 1.10
DEFAULT_LARGE_BOX_SPLIT_MIN_CLOSE_X_PX = 10.0
DEFAULT_EDGE_MODE = "strongest"


@dataclass(frozen=True)
class PipeEndPrediction:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    x_center: float
    y_center: float
    width: float
    height: float
    refined_x: float | None = None
    refined_y: float | None = None
    edge_strength: float | None = None
    postprocess_flags: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        payload = {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "box_xyxy": [float(self.x1), float(self.y1), float(self.x2), float(self.y2)],
            "box_xywh": [float(self.x_center), float(self.y_center), float(self.width), float(self.height)],
        }
        if self.refined_x is not None:
            payload["refined_pipe_end"] = {
                "x": float(self.refined_x),
                "y": float(self.refined_y if self.refined_y is not None else self.y_center),
                "edge_strength": None if self.edge_strength is None else float(self.edge_strength),
            }
        if self.postprocess_flags:
            payload["postprocess_flags"] = list(self.postprocess_flags)
        return payload


@dataclass(frozen=True)
class PipeEndInferenceResult:
    image_path: Path
    model_path: Path
    output_dir: Path
    predictions_path: Path
    overlay_path: Path
    image_width: int
    image_height: int
    predictions: list[PipeEndPrediction]
    imgsz: int
    conf: float
    iou: float
    device: str | None
    raw_prediction_count: int = 0
    postprocess: dict[str, Any] | None = None
    recovery_bounds_y: tuple[float, float] | None = None
    spacing_stats: dict[str, Any] | None = None

    @property
    def count(self) -> int:
        return len(self.predictions)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y", "enabled"}


def _float_env(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _int_env(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    if model_path:
        candidate = Path(model_path)
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"No existe el modelo YOLO pipe_end: {model_path}")

    env_path = os.environ.get("PIPE_END_YOLO_MODEL") or os.environ.get("TENARIS_PIPE_END_YOLO_MODEL")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"No existe PIPE_END_YOLO_MODEL: {env_path}")

    root = _repo_root()
    candidates = [
        root / "models" / "pipe_end_active" / "best.pt",
        root / "pipe_end_detection" / "models" / "pipe_end_active" / "best.pt",
        root.parent.parent / "pipe_end_yolo_project" / "models" / "pipe_end_active" / "best.pt",
        Path.home() / "Downloads" / "pipe_end_yolo_project" / "models" / "pipe_end_active" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "No se encontro modelo YOLO pipe_end. Configura PIPE_END_YOLO_MODEL o coloca best.pt en "
        f"models/pipe_end_active/best.pt.\nBuscado en:\n{searched}"
    )


@lru_cache(maxsize=4)
def _load_yolo_model(model_path: str):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Falta instalar ultralytics para usar deteccion YOLO pipe_end.") from exc
    return YOLO(model_path)


def _read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"No se pudo leer imagen para YOLO: {image_path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def _prediction_area(prediction: PipeEndPrediction) -> float:
    return max(0.0, float(prediction.x2 - prediction.x1)) * max(0.0, float(prediction.y2 - prediction.y1))


def _intersection_area(a: PipeEndPrediction, b: PipeEndPrediction) -> float:
    x1 = max(float(a.x1), float(b.x1))
    y1 = max(float(a.y1), float(b.y1))
    x2 = min(float(a.x2), float(b.x2))
    y2 = min(float(a.y2), float(b.y2))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_over_smaller(a: PipeEndPrediction, b: PipeEndPrediction) -> float:
    denom = min(_prediction_area(a), _prediction_area(b))
    if denom <= 1e-6:
        return 0.0
    return float(_intersection_area(a, b) / denom)


def _vertical_overlap_over_smaller(a: PipeEndPrediction, b: PipeEndPrediction) -> float:
    y1 = max(float(a.y1), float(b.y1))
    y2 = min(float(a.y2), float(b.y2))
    intersection = max(0.0, y2 - y1)
    denom = min(max(0.0, float(a.height)), max(0.0, float(b.height)))
    if denom <= 1e-6:
        return 0.0
    return float(intersection / denom)


def _copy_prediction(prediction: PipeEndPrediction, **updates: Any) -> PipeEndPrediction:
    data = {
        "class_id": prediction.class_id,
        "class_name": prediction.class_name,
        "confidence": prediction.confidence,
        "x1": prediction.x1,
        "y1": prediction.y1,
        "x2": prediction.x2,
        "y2": prediction.y2,
        "x_center": prediction.x_center,
        "y_center": prediction.y_center,
        "width": prediction.width,
        "height": prediction.height,
        "refined_x": prediction.refined_x,
        "refined_y": prediction.refined_y,
        "edge_strength": prediction.edge_strength,
        "postprocess_flags": prediction.postprocess_flags,
    }
    data.update(updates)
    return PipeEndPrediction(**data)


def _median(values: list[float]) -> float | None:
    clean = sorted(float(value) for value in values if float(value) > 1e-6)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float(0.5 * (clean[mid - 1] + clean[mid]))


def _postprocess_overlapping_predictions(
    predictions: list[PipeEndPrediction],
    *,
    overlap_threshold: float = DEFAULT_OVERLAP_SUPPRESSION,
    containment_ratio: float = DEFAULT_CONTAINMENT_RATIO,
    contained_children: int = DEFAULT_CONTAINED_CHILDREN,
    vertical_duplicate_height_ratio: float = DEFAULT_VERTICAL_DUPLICATE_HEIGHT_RATIO,
    vertical_duplicate_overlap: float = DEFAULT_VERTICAL_DUPLICATE_OVERLAP,
    vertical_duplicate_y_overlap: float = DEFAULT_VERTICAL_DUPLICATE_Y_OVERLAP,
) -> tuple[list[PipeEndPrediction], dict[str, Any]]:
    valid = [pred for pred in predictions if _prediction_area(pred) > 1.0]
    removed_containers: set[int] = set()
    for idx, candidate in enumerate(valid):
        candidate_area = _prediction_area(candidate)
        child_count = 0
        for other_idx, other in enumerate(valid):
            if idx == other_idx:
                continue
            other_area = _prediction_area(other)
            if other_area <= 1.0 or candidate_area <= other_area:
                continue
            if _intersection_area(candidate, other) / other_area >= float(containment_ratio):
                child_count += 1
        if child_count >= int(contained_children):
            removed_containers.add(idx)

    candidates = [pred for idx, pred in enumerate(valid) if idx not in removed_containers]
    ordered = sorted(candidates, key=lambda pred: (float(pred.confidence), -_prediction_area(pred)), reverse=True)
    keep: list[PipeEndPrediction] = []
    suppressed = 0
    for pred in ordered:
        if any(_overlap_over_smaller(pred, kept) > float(overlap_threshold) for kept in keep):
            suppressed += 1
            continue
        keep.append(pred)

    keep.sort(key=lambda pred: pred.y_center)
    vertical_suppressed = 0
    vertical_y_suppressed = 0
    if vertical_duplicate_height_ratio > 0 or vertical_duplicate_y_overlap > 0:
        deduped: list[PipeEndPrediction] = []
        for pred in keep:
            if deduped:
                previous = deduped[-1]
                vertical_gap = abs(float(pred.y_center) - float(previous.y_center))
                duplicate_gap = float(vertical_duplicate_height_ratio) * max(float(pred.height), float(previous.height))
                horizontal_gap = abs(float(pred.x_center) - float(previous.x_center))
                horizontal_limit = 1.25 * max(float(pred.width), float(previous.width))
                duplicate_overlap = _overlap_over_smaller(pred, previous)
                vertical_band_overlap = _vertical_overlap_over_smaller(pred, previous)
                is_local_duplicate = (
                    vertical_gap < duplicate_gap
                    or duplicate_overlap >= float(vertical_duplicate_overlap)
                )
                is_same_y_band = vertical_band_overlap >= float(vertical_duplicate_y_overlap)
                if (is_local_duplicate and horizontal_gap <= horizontal_limit) or is_same_y_band:
                    vertical_suppressed += 1
                    if is_same_y_band and not (is_local_duplicate and horizontal_gap <= horizontal_limit):
                        vertical_y_suppressed += 1
                    if float(pred.confidence) > float(previous.confidence):
                        deduped[-1] = _copy_prediction(
                            pred,
                            postprocess_flags=tuple(
                                [*pred.postprocess_flags, "kept_vertical_y_duplicate" if is_same_y_band else "kept_vertical_duplicate"]
                            ),
                        )
                    else:
                        deduped[-1] = _copy_prediction(
                            previous,
                            postprocess_flags=tuple(
                                [
                                    *previous.postprocess_flags,
                                    "kept_vertical_y_duplicate" if is_same_y_band else "kept_vertical_duplicate",
                                ]
                            ),
                        )
                    continue
            deduped.append(pred)
        keep = deduped

    return keep, {
        "raw_count": len(predictions),
        "valid_count": len(valid),
        "removed_large_containers": len(removed_containers),
        "suppressed_overlap": int(suppressed),
        "suppressed_vertical_duplicate": int(vertical_suppressed),
        "suppressed_vertical_y_duplicate": int(vertical_y_suppressed),
        "final_count_before_refine": len(keep),
        "overlap_threshold": float(overlap_threshold),
        "vertical_duplicate_height_ratio": float(vertical_duplicate_height_ratio),
        "vertical_duplicate_overlap": float(vertical_duplicate_overlap),
        "vertical_duplicate_y_overlap": float(vertical_duplicate_y_overlap),
        "containment_ratio": float(containment_ratio),
        "contained_children": int(contained_children),
    }


def _prediction_from_xyxy(
    coords: list[float] | tuple[float, float, float, float],
    confidence: float,
    *,
    flags: tuple[str, ...] = (),
) -> PipeEndPrediction:
    x1, y1, x2, y2 = [float(value) for value in coords]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return PipeEndPrediction(
        class_id=PIPE_END_CLASS_ID,
        class_name=PIPE_END_CLASS_NAME,
        confidence=float(confidence),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        x_center=x1 + 0.5 * width,
        y_center=y1 + 0.5 * height,
        width=width,
        height=height,
        postprocess_flags=flags,
    )


def _estimate_gap_recovery_pitch(predictions: list[PipeEndPrediction]) -> float | None:
    if len(predictions) < 2:
        return None
    ordered = sorted(predictions, key=lambda pred: float(pred.y_center))
    gaps = [
        abs(float(ordered[idx + 1].y_center) - float(ordered[idx].y_center))
        for idx in range(len(ordered) - 1)
    ]
    raw_gap = _median(gaps)
    heights = _median([float(pred.height) for pred in ordered])
    if raw_gap is None:
        return None if heights is None else 1.15 * float(heights)
    robust_gaps = [gap for gap in gaps if 0.45 * raw_gap <= gap <= 1.75 * raw_gap]
    pitch = _median(robust_gaps) or raw_gap
    if heights is not None:
        pitch = max(float(pitch), 1.10 * float(heights))
    return float(pitch)


def _filter_isolated_y_clusters(
    predictions: list[PipeEndPrediction],
    *,
    min_cluster_size: int,
    gap_ratio: float,
    min_gap_px: float,
    spacing_stats: dict[str, Any] | None = None,
) -> tuple[list[PipeEndPrediction], dict[str, Any]]:
    meta: dict[str, Any] = {
        "enabled": True,
        "input_count": len(predictions),
        "removed_count": 0,
        "kept_count": len(predictions),
        "cluster_count": 0,
        "min_cluster_size": int(min_cluster_size),
        "gap_ratio": float(gap_ratio),
        "min_gap_px": float(min_gap_px),
        "spacing_source": None,
    }
    if len(predictions) < max(4, int(min_cluster_size) + 1):
        return predictions, meta

    ordered = sorted(predictions, key=lambda pred: float(pred.y_center))
    median_height = _median([float(pred.height) for pred in ordered]) or 1.0
    gaps = [
        float(ordered[idx + 1].y_center) - float(ordered[idx].y_center)
        for idx in range(len(ordered) - 1)
        if float(ordered[idx + 1].y_center) > float(ordered[idx].y_center)
    ]
    if not gaps:
        return predictions, meta
    pitch = 0.0
    break_gap = 0.0
    if spacing_stats:
        try:
            pitch = float(spacing_stats.get("median_gap_px") or 0.0)
            break_gap = float(spacing_stats.get("allowed_gap_px") or 0.0)
        except (TypeError, ValueError):
            pitch = 0.0
            break_gap = 0.0
    if pitch > 1e-6 and break_gap > 1e-6:
        meta["spacing_source"] = spacing_stats.get("source", "annotation_spacing") if spacing_stats else None
        meta["annotation_spacing"] = {
            "median_gap_px": float(spacing_stats.get("median_gap_px", pitch)),
            "mean_gap_px": float(spacing_stats.get("mean_gap_px", 0.0)),
            "variance_gap_px": float(spacing_stats.get("variance_gap_px", 0.0)),
            "allowed_gap_px": float(break_gap),
        }
    else:
        local_gap_limit = max(float(min_gap_px), 4.0 * float(median_height))
        local_gaps = [gap for gap in gaps if gap <= local_gap_limit]
        pitch = _median(local_gaps) or _median(gaps) or (1.4 * float(median_height))
        break_gap = max(float(min_gap_px), float(gap_ratio) * float(pitch), 3.5 * float(median_height))
        meta["spacing_source"] = "current_image"
    meta["pitch"] = float(pitch)
    meta["break_gap"] = float(break_gap)

    clusters: list[list[PipeEndPrediction]] = [[ordered[0]]]
    previous_y = float(ordered[0].y_center)
    for pred in ordered[1:]:
        y_center = float(pred.y_center)
        if y_center - previous_y > break_gap:
            clusters.append([pred])
        else:
            clusters[-1].append(pred)
        previous_y = y_center
    meta["cluster_count"] = len(clusters)
    meta["cluster_sizes"] = [len(cluster) for cluster in clusters]
    if len(clusters) <= 1:
        return predictions, meta

    min_size = max(1, int(min_cluster_size))
    kept_clusters = [cluster for cluster in clusters if len(cluster) >= min_size]
    if not kept_clusters:
        kept_clusters = [max(clusters, key=len)]

    kept_ids = {id(pred) for cluster in kept_clusters for pred in cluster}
    kept = [pred for pred in ordered if id(pred) in kept_ids]
    meta["removed_count"] = len(predictions) - len(kept)
    meta["kept_count"] = len(kept)
    if meta["removed_count"] <= 0:
        return predictions, meta
    return kept, meta


def _filter_far_x_low_confidence_predictions(
    predictions: list[PipeEndPrediction],
    *,
    base_conf: float,
    distance_ratio: float,
    min_distance_px: float,
    min_conf: float,
    extra_conf: float,
) -> tuple[list[PipeEndPrediction], dict[str, Any]]:
    required_conf = max(float(min_conf), float(base_conf) + float(extra_conf))
    ordered = sorted(predictions, key=lambda pred: float(pred.y_center))
    median_width = _median([float(pred.width) for pred in ordered]) or 1.0
    distance_threshold = max(float(min_distance_px), float(distance_ratio) * float(median_width))
    meta: dict[str, Any] = {
        "enabled": True,
        "input_count": len(predictions),
        "removed_count": 0,
        "kept_count": len(predictions),
        "flagged_count": 0,
        "distance_threshold_px": float(distance_threshold),
        "required_conf": float(required_conf),
        "base_conf": float(base_conf),
        "distance_ratio": float(distance_ratio),
        "min_distance_px": float(min_distance_px),
        "min_conf": float(min_conf),
        "extra_conf": float(extra_conf),
    }
    if len(ordered) < 3:
        return predictions, meta

    kept: list[PipeEndPrediction] = []
    removed: list[dict[str, Any]] = []
    flagged = 0
    for idx, pred in enumerate(ordered):
        if idx == 0 or idx == len(ordered) - 1:
            kept.append(pred)
            continue
        upper = ordered[idx - 1]
        lower = ordered[idx + 1]
        y_span = float(lower.y_center) - float(upper.y_center)
        if abs(y_span) <= 1e-6:
            kept.append(pred)
            continue
        t = (float(pred.y_center) - float(upper.y_center)) / y_span
        t = max(0.0, min(1.0, t))
        expected_x = float(upper.x_center) + t * (float(lower.x_center) - float(upper.x_center))
        dx = float(pred.x_center) - expected_x
        if abs(dx) <= distance_threshold:
            kept.append(pred)
            continue
        flagged += 1
        if float(pred.confidence) >= required_conf:
            kept.append(
                _copy_prediction(
                    pred,
                    postprocess_flags=tuple([*pred.postprocess_flags, "kept_far_x_high_conf"]),
                )
            )
            continue
        removed.append(
            {
                "x_center": float(pred.x_center),
                "y_center": float(pred.y_center),
                "expected_x": float(expected_x),
                "dx": float(dx),
                "confidence": float(pred.confidence),
            }
        )

    meta["flagged_count"] = int(flagged)
    meta["removed_count"] = int(len(removed))
    meta["kept_count"] = int(len(kept))
    if removed:
        meta["removed"] = removed[:12]
        return kept, meta
    if flagged:
        return kept, meta
    return predictions, meta


def _gap_recovery_candidate_is_duplicate(
    candidate: PipeEndPrediction,
    existing: list[PipeEndPrediction],
    *,
    expected_pitch: float,
) -> bool:
    for pred in existing:
        if _overlap_over_smaller(candidate, pred) >= 0.20:
            return True
        if _vertical_overlap_over_smaller(candidate, pred) >= 0.70:
            return True
        if abs(float(candidate.y_center) - float(pred.y_center)) <= 0.35 * float(expected_pitch):
            return True
    return False


def _predict_gap_candidates_in_band(
    yolo: Any,
    image: np.ndarray,
    predict_kwargs: dict[str, Any],
    *,
    y0: int,
    y1: int,
    target_y: float,
    expected_pitch: float,
    recovery_conf: float,
    existing: list[PipeEndPrediction],
    flag: str,
) -> list[PipeEndPrediction]:
    image_h, image_w = image.shape[:2]
    y0 = max(0, min(image_h, int(y0)))
    y1 = max(0, min(image_h, int(y1)))
    if y1 <= y0 + 4:
        return []
    crop = image[y0:y1, 0:image_w]
    crop_kwargs = dict(predict_kwargs)
    crop_kwargs["conf"] = float(recovery_conf)
    crop_kwargs["max_det"] = min(int(crop_kwargs.get("max_det", 256)), 48)
    raw_results = yolo.predict(crop, **crop_kwargs)
    candidates: list[PipeEndPrediction] = []
    if not raw_results:
        return candidates
    result = raw_results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return candidates
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    for class_id, coords, score in zip(cls, xyxy, scores):
        if int(class_id) != PIPE_END_CLASS_ID:
            continue
        x1, local_y1, x2, local_y2 = [float(value) for value in coords]
        candidate = _prediction_from_xyxy(
            [x1, local_y1 + y0, x2, local_y2 + y0],
            float(score),
            flags=(flag,),
        )
        if abs(float(candidate.y_center) - float(target_y)) > 0.65 * float(expected_pitch):
            continue
        if _gap_recovery_candidate_is_duplicate(candidate, existing, expected_pitch=expected_pitch):
            continue
        candidates.append(candidate)
    return candidates


def _synthesize_pitch_candidate(
    target_y: float,
    references: list[PipeEndPrediction],
    *,
    image_width: int,
    image_height: int,
    expected_pitch: float,
    confidence: float,
    flag: str,
) -> PipeEndPrediction | None:
    if not references:
        return None
    ordered = sorted(references, key=lambda pred: float(pred.y_center))
    above = [pred for pred in ordered if float(pred.y_center) <= float(target_y)]
    below = [pred for pred in ordered if float(pred.y_center) >= float(target_y)]
    top = above[-1] if above else ordered[0]
    bottom = below[0] if below else ordered[-1]
    if top is bottom:
        ref_x_center = float(top.x_center)
        ref_width = float(top.width)
        ref_height = float(top.height)
    else:
        denom = max(1e-6, float(bottom.y_center) - float(top.y_center))
        ratio = max(0.0, min(1.0, (float(target_y) - float(top.y_center)) / denom))
        ref_x_center = float(top.x_center) + ratio * (float(bottom.x_center) - float(top.x_center))
        ref_width = float(top.width) + ratio * (float(bottom.width) - float(top.width))
        ref_height = float(top.height) + ratio * (float(bottom.height) - float(top.height))

    median_width = _median([float(pred.width) for pred in references]) or ref_width
    median_height = _median([float(pred.height) for pred in references]) or ref_height
    width = max(6.0, min(2.0 * float(median_width), max(float(ref_width), 0.80 * float(median_width))))
    height = max(6.0, min(1.30 * float(median_height), max(float(ref_height), 0.80 * float(median_height))))
    if height > 0.90 * float(expected_pitch):
        height = max(6.0, 0.75 * float(expected_pitch))
    x1 = max(0.0, min(float(image_width) - 2.0, ref_x_center - 0.5 * width))
    x2 = min(float(image_width), x1 + width)
    y1 = max(0.0, min(float(image_height) - 2.0, float(target_y) - 0.5 * height))
    y2 = min(float(image_height), y1 + height)
    if x2 <= x1 + 2.0 or y2 <= y1 + 2.0:
        return None
    return _prediction_from_xyxy([x1, y1, x2, y2], float(confidence), flags=(flag,))


def _recover_gap_predictions(
    image_path: Path,
    yolo: Any,
    predictions: list[PipeEndPrediction],
    predict_kwargs: dict[str, Any],
    *,
    recovery_conf: float,
    min_gap_ratio: float,
    max_gap_ratio: float,
    empty_gap_ratio: float,
    band_ratio: float,
    max_missing_per_gap: int,
    pitch_fallback_enabled: bool,
    pitch_fallback_conf: float,
) -> tuple[list[PipeEndPrediction], dict[str, Any]]:
    meta: dict[str, Any] = {
        "enabled": True,
        "attempted_gaps": 0,
        "candidate_count": 0,
        "recovered_count": 0,
        "pitch_fallback_count": 0,
        "conf": float(recovery_conf),
        "min_gap_ratio": float(min_gap_ratio),
        "max_gap_ratio": float(max_gap_ratio),
        "empty_gap_ratio": float(empty_gap_ratio),
        "band_ratio": float(band_ratio),
        "pitch_fallback_enabled": bool(pitch_fallback_enabled),
    }
    if len(predictions) < 2:
        return [], meta

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return [], meta
    image_h, image_w = image.shape[:2]
    ordered = sorted(predictions, key=lambda pred: float(pred.y_center))
    fallback_pitch = _estimate_gap_recovery_pitch(ordered)
    if fallback_pitch is None or fallback_pitch <= 1e-6:
        return [], meta

    recovered: list[PipeEndPrediction] = []
    max_missing = max(1, int(max_missing_per_gap))
    for lower_idx in range(len(ordered) - 1):
        top_pred = ordered[lower_idx]
        bottom_pred = ordered[lower_idx + 1]
        gap = abs(float(bottom_pred.y_center) - float(top_pred.y_center))
        local_height = _median([float(top_pred.height), float(bottom_pred.height)]) or fallback_pitch
        expected_pitch = max(float(fallback_pitch), 1.10 * float(local_height))
        if expected_pitch <= 1e-6:
            continue
        gap_ratio = gap / expected_pitch
        empty_gap = max(0.0, float(bottom_pred.y1) - float(top_pred.y2))
        empty_gap_trigger = empty_gap / max(1.0, float(local_height)) >= float(empty_gap_ratio)
        if gap_ratio > float(max_gap_ratio):
            continue
        if gap_ratio < float(min_gap_ratio) and not empty_gap_trigger:
            continue
        missing_count = min(max_missing, max(1, int(round(gap_ratio)) - 1))
        if empty_gap_trigger:
            missing_count = max(1, missing_count)
        for missing_idx in range(missing_count):
            if empty_gap_trigger and missing_count == 1:
                target_y = 0.5 * (float(top_pred.y2) + float(bottom_pred.y1))
            else:
                target_y = float(top_pred.y_center) + (missing_idx + 1) * gap / (missing_count + 1)
            band_half = max(8.0, 0.5 * float(band_ratio) * expected_pitch)
            y0 = max(0, int(round(target_y - band_half)))
            y1 = min(image_h, int(round(target_y + band_half)))
            if y1 <= y0 + 4:
                continue
            meta["attempted_gaps"] += 1
            candidates = _predict_gap_candidates_in_band(
                yolo,
                image,
                predict_kwargs,
                y0=y0,
                y1=y1,
                target_y=target_y,
                expected_pitch=expected_pitch,
                recovery_conf=recovery_conf,
                existing=[*ordered, *recovered],
                flag="gap_recovered",
            )
            meta["candidate_count"] += len(candidates)
            if candidates:
                recovered.append(max(candidates, key=lambda pred: float(pred.confidence)))
            elif pitch_fallback_enabled:
                fallback = _synthesize_pitch_candidate(
                    target_y,
                    [top_pred, bottom_pred, *ordered],
                    image_width=image_w,
                    image_height=image_h,
                    expected_pitch=expected_pitch,
                    confidence=float(pitch_fallback_conf),
                    flag="gap_pitch_fallback",
                )
                if fallback is not None and not _gap_recovery_candidate_is_duplicate(
                    fallback,
                    [*ordered, *recovered],
                    expected_pitch=expected_pitch,
                ):
                    recovered.append(fallback)
                    meta["pitch_fallback_count"] += 1

    meta["recovered_count"] = len(recovered)
    return recovered, meta


def _recover_edge_gap_predictions(
    image_path: Path,
    yolo: Any,
    predictions: list[PipeEndPrediction],
    predict_kwargs: dict[str, Any],
    *,
    recovery_conf: float,
    bounds_y: tuple[float, float] | None,
    min_gap_ratio: float,
    edge_space_ratio: float,
    band_ratio: float,
    max_missing_per_edge: int,
    pitch_fallback_enabled: bool,
    pitch_fallback_conf: float,
) -> tuple[list[PipeEndPrediction], dict[str, Any]]:
    meta: dict[str, Any] = {
        "enabled": True,
        "attempted_edges": 0,
        "candidate_count": 0,
        "recovered_count": 0,
        "pitch_fallback_count": 0,
        "edge_space_attempts": 0,
        "conf": float(recovery_conf),
        "min_gap_ratio": float(min_gap_ratio),
        "edge_space_ratio": float(edge_space_ratio),
        "band_ratio": float(band_ratio),
        "bounds_y": None if bounds_y is None else [float(bounds_y[0]), float(bounds_y[1])],
        "pitch_fallback_enabled": bool(pitch_fallback_enabled),
    }
    if not predictions:
        return [], meta
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return [], meta
    image_h = image.shape[0]
    ordered = sorted(predictions, key=lambda pred: float(pred.y_center))
    expected_pitch = _estimate_gap_recovery_pitch(ordered)
    if expected_pitch is None or expected_pitch <= 1e-6:
        return [], meta
    if bounds_y is None:
        return [], meta
    top_bound = max(0.0, min(float(image_h), float(bounds_y[0])))
    bottom_bound = max(0.0, min(float(image_h), float(bounds_y[1])))
    if bottom_bound <= top_bound + expected_pitch:
        return [], meta

    median_height = _median([float(pred.height) for pred in ordered]) or max(1.0, 0.75 * float(expected_pitch))
    recovered: list[PipeEndPrediction] = []
    max_missing = max(1, int(max_missing_per_edge))

    def add_target(targets: list[float], target_y: float) -> None:
        if not np.isfinite(float(target_y)):
            return
        if all(abs(float(target_y) - existing_y) >= 0.45 * float(median_height) for existing_y in targets):
            targets.append(float(target_y))

    def recover_side(*, anchor: PipeEndPrediction, direction: int, limit_y: float, flag: str) -> None:
        nonlocal recovered
        anchor_y = float(anchor.y_center)
        anchor_edge = float(anchor.y1) if direction < 0 else float(anchor.y2)
        center_gap = (anchor_y - limit_y) if direction < 0 else (limit_y - anchor_y)
        edge_space = (anchor_edge - limit_y) if direction < 0 else (limit_y - anchor_edge)
        targets: list[float] = []

        if edge_space / max(1.0, float(median_height)) >= float(edge_space_ratio):
            meta["edge_space_attempts"] += 1
            add_target(targets, limit_y - direction * 0.5 * float(median_height))

        if center_gap / expected_pitch >= float(min_gap_ratio):
            missing_count = min(max_missing, max(1, int(round(center_gap / expected_pitch))))
            for missing_idx in range(missing_count):
                target_y = anchor_y + direction * expected_pitch * float(missing_idx + 1)
                if direction < 0 and target_y < limit_y + 0.30 * float(median_height):
                    break
                if direction > 0 and target_y > limit_y - 0.30 * float(median_height):
                    break
                add_target(targets, target_y)

        for target_y in targets:
            band_half = max(8.0, 0.5 * float(band_ratio) * expected_pitch)
            y0 = int(round(target_y - band_half))
            y1 = int(round(target_y + band_half))
            meta["attempted_edges"] += 1
            candidates = _predict_gap_candidates_in_band(
                yolo,
                image,
                predict_kwargs,
                y0=y0,
                y1=y1,
                target_y=target_y,
                expected_pitch=expected_pitch,
                recovery_conf=recovery_conf,
                existing=[*ordered, *recovered],
                flag=flag,
            )
            meta["candidate_count"] += len(candidates)
            if candidates:
                recovered.append(max(candidates, key=lambda pred: float(pred.confidence)))
            elif pitch_fallback_enabled:
                fallback = _synthesize_pitch_candidate(
                    target_y,
                    ordered,
                    image_width=image.shape[1],
                    image_height=image_h,
                    expected_pitch=expected_pitch,
                    confidence=float(pitch_fallback_conf),
                    flag=f"{flag}_pitch_fallback",
                )
                if fallback is not None and not _gap_recovery_candidate_is_duplicate(
                    fallback,
                    [*ordered, *recovered],
                    expected_pitch=expected_pitch,
                ):
                    recovered.append(fallback)
                    meta["pitch_fallback_count"] += 1

    recover_side(anchor=ordered[0], direction=-1, limit_y=top_bound, flag="edge_gap_top_recovered")
    recover_side(anchor=ordered[-1], direction=1, limit_y=bottom_bound, flag="edge_gap_bottom_recovered")
    meta["recovered_count"] = len(recovered)
    return recovered, meta


def _smooth_profile(profile: np.ndarray) -> np.ndarray:
    if profile.size < 5:
        return profile.astype(np.float32, copy=False)
    kernel = min(9, int(profile.size) if int(profile.size) % 2 == 1 else int(profile.size) - 1)
    kernel = max(3, kernel)
    return cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32), (kernel, 1), 0).reshape(-1)


def _pick_sobel_y_separators(
    profile: np.ndarray,
    *,
    wanted: int,
    min_separation: float,
    min_strength_ratio: float,
) -> list[int]:
    if wanted <= 0 or profile.size < 5 or float(np.max(profile)) <= 1e-6:
        return []
    max_strength = float(np.max(profile))
    threshold = max(float(min_strength_ratio) * max_strength, float(np.percentile(profile, 70)))
    margin = max(2, int(round(0.18 * float(profile.size))))
    candidate_rows: list[tuple[float, int]] = []
    for idx in range(margin, int(profile.size) - margin):
        value = float(profile[idx])
        if value < threshold:
            continue
        left = float(profile[idx - 1])
        right = float(profile[idx + 1])
        if value < left or value < right:
            continue
        candidate_rows.append((value, idx))
    selected: list[int] = []
    for _, row in sorted(candidate_rows, reverse=True):
        if all(abs(row - prev) >= float(min_separation) for prev in selected):
            selected.append(int(row))
        if len(selected) >= wanted:
            break
    selected.sort()
    return selected


def _split_large_predictions_with_sobel_y(
    image_path: Path,
    predictions: list[PipeEndPrediction],
    *,
    height_ratio: float,
    max_parts: int,
    min_edge_strength_ratio: float,
    close_x_ratio: float,
    min_close_x_px: float,
) -> tuple[list[PipeEndPrediction], dict[str, Any]]:
    meta: dict[str, Any] = {
        "enabled": True,
        "checked_count": 0,
        "candidate_count": 0,
        "sobel_checked_count": 0,
        "split_count": 0,
        "added_count": 0,
        "skipped_far_left_lower_neighbor": 0,
        "skipped_far_x_lower_neighbor": 0,
        "height_ratio": float(height_ratio),
        "max_parts": int(max_parts),
        "min_edge_strength_ratio": float(min_edge_strength_ratio),
        "close_x_ratio": float(close_x_ratio),
        "min_close_x_px": float(min_close_x_px),
    }
    if len(predictions) < 2:
        return predictions, meta
    median_height = _median([float(pred.height) for pred in predictions])
    if median_height is None or median_height <= 1e-6:
        return predictions, meta
    median_width = _median([float(pred.width) for pred in predictions]) or 0.0
    ordered = sorted(predictions, key=lambda pred: float(pred.y_center))
    ordered_index = {id(pred): idx for idx, pred in enumerate(ordered)}
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return predictions, meta
    image_h, image_w = image.shape[:2]
    split_predictions: list[PipeEndPrediction] = []
    for pred in predictions:
        idx = ordered_index.get(id(pred))
        upper_neighbor = ordered[idx - 1] if idx is not None and idx > 0 else None
        lower_neighbor = ordered[idx + 1] if idx is not None and idx + 1 < len(ordered) else None
        local_reference_height = _median(
            [
                float(neighbor.height)
                for neighbor in (upper_neighbor, lower_neighbor)
                if neighbor is not None and float(neighbor.height) > 1e-6
            ]
        )
        reference_height = float(local_reference_height or median_height)
        if float(pred.height) < float(height_ratio) * reference_height:
            split_predictions.append(pred)
            continue
        meta["candidate_count"] += 1
        close_x_threshold = max(
            float(min_close_x_px),
            float(close_x_ratio) * max(float(median_width), 1.0),
        )
        if lower_neighbor is not None:
            lower_dx = float(lower_neighbor.x_center) - float(pred.x_center)
            if lower_dx < -close_x_threshold:
                meta["skipped_far_left_lower_neighbor"] += 1
                split_predictions.append(pred)
                continue
            if abs(lower_dx) > close_x_threshold:
                meta["skipped_far_x_lower_neighbor"] += 1
                split_predictions.append(pred)
                continue
        meta["checked_count"] += 1
        meta["sobel_checked_count"] += 1
        estimated_parts = min(max(2, int(round(float(pred.height) / float(median_height)))), max(2, int(max_parts)))
        wanted_separators = max(1, estimated_parts - 1)
        pad_x = max(2, int(round(0.15 * max(1.0, float(pred.width)))))
        x0 = max(0, int(np.floor(pred.x1)) - pad_x)
        x1 = min(image_w, int(np.ceil(pred.x2)) + pad_x)
        y0 = max(0, int(np.floor(pred.y1)))
        y1 = min(image_h, int(np.ceil(pred.y2)))
        roi = image[y0:y1, x0:x1]
        if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 3:
            split_predictions.append(pred)
            continue
        blurred = cv2.GaussianBlur(roi, (1, 5), 0)
        grad_y = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        profile = np.percentile(np.abs(grad_y), 90, axis=1).astype(np.float32)
        profile = _smooth_profile(profile)
        separators = _pick_sobel_y_separators(
            profile,
            wanted=wanted_separators,
            min_separation=max(3.0, 0.55 * float(median_height)),
            min_strength_ratio=float(min_edge_strength_ratio),
        )
        if len(separators) < wanted_separators:
            split_predictions.append(pred)
            continue
        boundaries = [float(pred.y1), *[float(y0 + sep) for sep in separators[:wanted_separators]], float(pred.y2)]
        parts: list[PipeEndPrediction] = []
        valid_parts = True
        for idx in range(len(boundaries) - 1):
            part_y1 = max(0.0, min(float(image_h), boundaries[idx]))
            part_y2 = max(0.0, min(float(image_h), boundaries[idx + 1]))
            part_h = part_y2 - part_y1
            if part_h < 0.35 * float(median_height):
                valid_parts = False
                break
            parts.append(
                _prediction_from_xyxy(
                    [float(pred.x1), part_y1, float(pred.x2), part_y2],
                    max(0.01, 0.92 * float(pred.confidence)),
                    flags=tuple([*pred.postprocess_flags, "split_sobel_y"]),
                )
            )
        if valid_parts and len(parts) >= 2:
            split_predictions.extend(parts)
            meta["split_count"] += 1
            meta["added_count"] += len(parts) - 1
        else:
            split_predictions.append(pred)
    return split_predictions, meta


def _refine_prediction_with_sobel_x(
    gray: np.ndarray,
    prediction: PipeEndPrediction,
    *,
    edge_mode: str = DEFAULT_EDGE_MODE,
) -> PipeEndPrediction:
    height, width = gray.shape[:2]
    pad_x = max(2, int(round(0.18 * max(1.0, prediction.width))))
    pad_y = max(1, int(round(0.12 * max(1.0, prediction.height))))
    x0 = max(0, int(np.floor(prediction.x1)) - pad_x)
    x1 = min(width, int(np.ceil(prediction.x2)) + pad_x)
    y0 = max(0, int(np.floor(prediction.y1)) - pad_y)
    y1 = min(height, int(np.ceil(prediction.y2)) + pad_y)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0 or roi.shape[1] < 3 or roi.shape[0] < 2:
        return _copy_prediction(
            prediction,
            refined_x=float(prediction.x_center),
            refined_y=float(prediction.y_center),
            edge_strength=0.0,
            postprocess_flags=tuple([*prediction.postprocess_flags, "sobel_fallback_center"]),
        )

    blur_kernel = (5, 1) if roi.shape[1] >= 5 else (3, 1)
    blurred = cv2.GaussianBlur(roi, blur_kernel, 0)
    grad_x = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    abs_grad = np.abs(grad_x)
    profile = np.percentile(abs_grad, 90, axis=0).astype(np.float32)
    profile = _smooth_profile(profile)
    if profile.size == 0 or float(np.max(profile)) <= 1e-6:
        return _copy_prediction(
            prediction,
            refined_x=float(prediction.x_center),
            refined_y=float(prediction.y_center),
            edge_strength=0.0,
            postprocess_flags=tuple([*prediction.postprocess_flags, "sobel_no_edge"]),
        )

    peak_threshold = max(0.35 * float(np.max(profile)), float(np.percentile(profile, 75)))
    candidate_cols = np.flatnonzero(profile >= peak_threshold)
    mode = str(edge_mode or DEFAULT_EDGE_MODE).strip().lower()
    if candidate_cols.size and mode == "leftmost":
        local_x = int(candidate_cols[0])
    elif candidate_cols.size and mode == "rightmost":
        local_x = int(candidate_cols[-1])
    else:
        local_x = int(np.argmax(profile))

    refined_x = float(x0 + local_x)
    return _copy_prediction(
        prediction,
        refined_x=refined_x,
        refined_y=float(prediction.y_center),
        edge_strength=float(profile[local_x]),
        postprocess_flags=tuple([*prediction.postprocess_flags, f"sobel_x_{mode or DEFAULT_EDGE_MODE}"]),
    )


def _refine_predictions_with_sobel_x(
    image_path: Path,
    predictions: list[PipeEndPrediction],
    *,
    edge_mode: str = DEFAULT_EDGE_MODE,
) -> list[PipeEndPrediction]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"No se pudo leer imagen para refinamiento Sobel-X: {image_path}")
    return [_refine_prediction_with_sobel_x(image, pred, edge_mode=edge_mode) for pred in predictions]


def _draw_overlay(image_path: Path, predictions: list[PipeEndPrediction], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"No se pudo leer imagen para overlay YOLO: {image_path}")

    ordered = sorted(predictions, key=lambda pred: pred.y_center)
    for pos, pred in enumerate(ordered):
        tube_idx = int(pos + 1)
        p1 = (int(round(pred.x1)), int(round(pred.y1)))
        p2 = (int(round(pred.x2)), int(round(pred.y2)))
        center = (int(round(pred.x_center)), int(round(pred.y_center)))
        refined_x = float(pred.refined_x if pred.refined_x is not None else pred.x_center)
        refined_y = float(pred.refined_y if pred.refined_y is not None else pred.y_center)
        refined = (int(round(refined_x)), int(round(refined_y)))
        cv2.rectangle(image, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(image, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.line(image, (refined[0], p1[1]), (refined[0], p2[1]), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(image, refined, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{tube_idx} {pred.confidence:.2f}",
            (p1[0] + 4, max(14, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def predictions_to_x_start_list(predictions: list[PipeEndPrediction]) -> list[dict[str, Any]]:
    """Convert YOLO boxes to the notebook pipeline's x_start_list format.

    Tube numbering is top-to-bottom: the highest visible pipe is tube 1.
    """
    ordered = sorted(predictions, key=lambda pred: pred.y_center)
    x_start_list: list[dict[str, Any]] = []
    for pos, pred in enumerate(ordered):
        tube_idx = int(pos + 1)
        x_start = float(pred.refined_x if pred.refined_x is not None else pred.x_center)
        y_center = float(pred.refined_y if pred.refined_y is not None else pred.y_center)
        x_start_list.append(
            {
                "tube_idx": tube_idx,
                "x_start": x_start,
                "x_local": x_start,
                "x_end_estimate": x_start,
                "x_seed": x_start,
                "x_start_smooth": x_start,
                "y_center": y_center,
                "confidence": float(pred.confidence),
                "source": "yolo_pipe_end",
                "box_xyxy": [float(pred.x1), float(pred.y1), float(pred.x2), float(pred.y2)],
                "box_xywh": [float(pred.x_center), float(pred.y_center), float(pred.width), float(pred.height)],
                "yolo_box_center": [float(pred.x_center), float(pred.y_center)],
                "refined_pipe_end": {
                    "x": x_start,
                    "y": y_center,
                    "edge_strength": None if pred.edge_strength is None else float(pred.edge_strength),
                },
            }
        )
    x_start_list.sort(key=lambda item: int(item["tube_idx"]))
    return x_start_list


def run_pipe_end_inference(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    device: str | None = None,
    recovery_bounds_y: tuple[float, float] | None = None,
    spacing_stats: dict[str, Any] | None = None,
) -> PipeEndInferenceResult:
    image = Path(image_path).resolve()
    model = resolve_model_path(model_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    image_width, image_height = _read_image_size(image)
    yolo = _load_yolo_model(str(model))
    predict_kwargs: dict[str, Any] = {
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "verbose": False,
        "max_det": 256,
    }
    if device:
        predict_kwargs["device"] = str(device)

    raw_results = yolo.predict(str(image), **predict_kwargs)
    predictions: list[PipeEndPrediction] = []
    if raw_results:
        result = raw_results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            for class_id, coords, score in zip(cls, xyxy, scores):
                class_int = int(class_id)
                if class_int != PIPE_END_CLASS_ID:
                    continue
                x1, y1, x2, y2 = [float(value) for value in coords]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                predictions.append(
                    PipeEndPrediction(
                        class_id=class_int,
                        class_name=PIPE_END_CLASS_NAME,
                        confidence=float(score),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        x_center=x1 + 0.5 * width,
                        y_center=y1 + 0.5 * height,
                        width=width,
                        height=height,
                    )
                )

    raw_prediction_count = len(predictions)
    overlap_threshold = _float_env("PIPE_END_YOLO_OVERLAP_THRESHOLD", DEFAULT_OVERLAP_SUPPRESSION)
    containment_ratio = _float_env("PIPE_END_YOLO_CONTAINMENT_RATIO", DEFAULT_CONTAINMENT_RATIO)
    contained_children = _int_env("PIPE_END_YOLO_CONTAINED_CHILDREN", DEFAULT_CONTAINED_CHILDREN)
    vertical_duplicate_height_ratio = _float_env(
        "PIPE_END_YOLO_VERTICAL_DUPLICATE_HEIGHT_RATIO", DEFAULT_VERTICAL_DUPLICATE_HEIGHT_RATIO
    )
    vertical_duplicate_overlap = _float_env("PIPE_END_YOLO_VERTICAL_DUPLICATE_OVERLAP", DEFAULT_VERTICAL_DUPLICATE_OVERLAP)
    vertical_duplicate_y_overlap = _float_env(
        "PIPE_END_YOLO_VERTICAL_DUPLICATE_Y_OVERLAP", DEFAULT_VERTICAL_DUPLICATE_Y_OVERLAP
    )
    edge_mode = str(os.environ.get("PIPE_END_YOLO_EDGE_MODE", DEFAULT_EDGE_MODE)).strip().lower() or DEFAULT_EDGE_MODE
    predictions, postprocess = _postprocess_overlapping_predictions(
        predictions,
        overlap_threshold=overlap_threshold,
        containment_ratio=containment_ratio,
        contained_children=contained_children,
        vertical_duplicate_height_ratio=vertical_duplicate_height_ratio,
        vertical_duplicate_overlap=vertical_duplicate_overlap,
        vertical_duplicate_y_overlap=vertical_duplicate_y_overlap,
    )
    isolated_filter_meta: dict[str, Any] = {"enabled": False}
    if _bool_env("PIPE_END_YOLO_ISOLATED_FILTER_ENABLED", DEFAULT_ISOLATED_FILTER_ENABLED):
        predictions, isolated_filter_meta = _filter_isolated_y_clusters(
            predictions,
            min_cluster_size=_int_env(
                "PIPE_END_YOLO_ISOLATED_FILTER_MIN_CLUSTER_SIZE",
                DEFAULT_ISOLATED_FILTER_MIN_CLUSTER_SIZE,
            ),
            gap_ratio=_float_env("PIPE_END_YOLO_ISOLATED_FILTER_GAP_RATIO", DEFAULT_ISOLATED_FILTER_GAP_RATIO),
            min_gap_px=_float_env("PIPE_END_YOLO_ISOLATED_FILTER_MIN_GAP_PX", DEFAULT_ISOLATED_FILTER_MIN_GAP_PX),
            spacing_stats=spacing_stats,
        )
    gap_recovery_meta: dict[str, Any] = {"enabled": False}
    edge_gap_recovery_meta: dict[str, Any] = {"enabled": False}
    if _bool_env("PIPE_END_YOLO_GAP_RECOVERY_ENABLED", DEFAULT_GAP_RECOVERY_ENABLED):
        recovery_conf = _float_env("PIPE_END_YOLO_GAP_RECOVERY_CONF", min(DEFAULT_GAP_RECOVERY_CONF, float(conf)))
        recovered, gap_recovery_meta = _recover_gap_predictions(
            image,
            yolo,
            predictions,
            predict_kwargs,
            recovery_conf=recovery_conf,
            min_gap_ratio=_float_env("PIPE_END_YOLO_GAP_RECOVERY_MIN_RATIO", DEFAULT_GAP_RECOVERY_MIN_RATIO),
            max_gap_ratio=_float_env("PIPE_END_YOLO_GAP_RECOVERY_MAX_RATIO", DEFAULT_GAP_RECOVERY_MAX_RATIO),
            empty_gap_ratio=_float_env(
                "PIPE_END_YOLO_GAP_RECOVERY_EMPTY_GAP_RATIO", DEFAULT_GAP_RECOVERY_EMPTY_GAP_RATIO
            ),
            band_ratio=_float_env("PIPE_END_YOLO_GAP_RECOVERY_BAND_RATIO", DEFAULT_GAP_RECOVERY_BAND_RATIO),
            max_missing_per_gap=_int_env(
                "PIPE_END_YOLO_GAP_RECOVERY_MAX_MISSING_PER_GAP", DEFAULT_GAP_RECOVERY_MAX_MISSING_PER_GAP
            ),
            pitch_fallback_enabled=_bool_env(
                "PIPE_END_YOLO_GAP_RECOVERY_PITCH_FALLBACK_ENABLED",
                DEFAULT_GAP_RECOVERY_PITCH_FALLBACK_ENABLED,
            ),
            pitch_fallback_conf=_float_env(
                "PIPE_END_YOLO_GAP_RECOVERY_PITCH_FALLBACK_CONF",
                DEFAULT_GAP_RECOVERY_PITCH_FALLBACK_CONF,
            ),
        )
        if recovered:
            predictions, gap_postprocess = _postprocess_overlapping_predictions(
                [*predictions, *recovered],
                overlap_threshold=overlap_threshold,
                containment_ratio=containment_ratio,
                contained_children=contained_children,
                vertical_duplicate_height_ratio=vertical_duplicate_height_ratio,
                vertical_duplicate_overlap=vertical_duplicate_overlap,
                vertical_duplicate_y_overlap=vertical_duplicate_y_overlap,
            )
            gap_recovery_meta["postprocess"] = gap_postprocess
        if _bool_env("PIPE_END_YOLO_EDGE_GAP_RECOVERY_ENABLED", DEFAULT_EDGE_GAP_RECOVERY_ENABLED):
            edge_recovered, edge_gap_recovery_meta = _recover_edge_gap_predictions(
                image,
                yolo,
                predictions,
                predict_kwargs,
                recovery_conf=recovery_conf,
                bounds_y=recovery_bounds_y,
                min_gap_ratio=_float_env(
                    "PIPE_END_YOLO_EDGE_GAP_RECOVERY_MIN_RATIO", DEFAULT_EDGE_GAP_RECOVERY_MIN_RATIO
                ),
                edge_space_ratio=_float_env(
                    "PIPE_END_YOLO_EDGE_GAP_RECOVERY_EDGE_SPACE_RATIO",
                    DEFAULT_EDGE_GAP_RECOVERY_EDGE_SPACE_RATIO,
                ),
                band_ratio=_float_env("PIPE_END_YOLO_GAP_RECOVERY_BAND_RATIO", DEFAULT_GAP_RECOVERY_BAND_RATIO),
                max_missing_per_edge=_int_env(
                    "PIPE_END_YOLO_EDGE_GAP_RECOVERY_MAX_MISSING_PER_EDGE",
                    DEFAULT_EDGE_GAP_RECOVERY_MAX_MISSING_PER_EDGE,
                ),
                pitch_fallback_enabled=_bool_env(
                    "PIPE_END_YOLO_GAP_RECOVERY_PITCH_FALLBACK_ENABLED",
                    DEFAULT_GAP_RECOVERY_PITCH_FALLBACK_ENABLED,
                ),
                pitch_fallback_conf=_float_env(
                    "PIPE_END_YOLO_GAP_RECOVERY_PITCH_FALLBACK_CONF",
                    DEFAULT_GAP_RECOVERY_PITCH_FALLBACK_CONF,
                ),
            )
            if edge_recovered:
                predictions, edge_gap_postprocess = _postprocess_overlapping_predictions(
                    [*predictions, *edge_recovered],
                    overlap_threshold=overlap_threshold,
                    containment_ratio=containment_ratio,
                    contained_children=contained_children,
                    vertical_duplicate_height_ratio=vertical_duplicate_height_ratio,
                    vertical_duplicate_overlap=vertical_duplicate_overlap,
                    vertical_duplicate_y_overlap=vertical_duplicate_y_overlap,
                )
                edge_gap_recovery_meta["postprocess"] = edge_gap_postprocess
    far_x_conf_filter_meta: dict[str, Any] = {"enabled": False}
    if _bool_env("PIPE_END_YOLO_FAR_X_CONF_FILTER_ENABLED", DEFAULT_FAR_X_CONF_FILTER_ENABLED):
        predictions, far_x_conf_filter_meta = _filter_far_x_low_confidence_predictions(
            predictions,
            base_conf=float(conf),
            distance_ratio=_float_env(
                "PIPE_END_YOLO_FAR_X_CONF_FILTER_DISTANCE_RATIO",
                DEFAULT_FAR_X_CONF_FILTER_DISTANCE_RATIO,
            ),
            min_distance_px=_float_env(
                "PIPE_END_YOLO_FAR_X_CONF_FILTER_MIN_DISTANCE_PX",
                DEFAULT_FAR_X_CONF_FILTER_MIN_DISTANCE_PX,
            ),
            min_conf=_float_env(
                "PIPE_END_YOLO_FAR_X_CONF_FILTER_MIN_CONF",
                DEFAULT_FAR_X_CONF_FILTER_MIN_CONF,
            ),
            extra_conf=_float_env(
                "PIPE_END_YOLO_FAR_X_CONF_FILTER_EXTRA_CONF",
                DEFAULT_FAR_X_CONF_FILTER_EXTRA_CONF,
            ),
        )
    large_box_split_meta: dict[str, Any] = {"enabled": False}
    if _bool_env("PIPE_END_YOLO_LARGE_BOX_SPLIT_ENABLED", DEFAULT_LARGE_BOX_SPLIT_ENABLED):
        predictions, large_box_split_meta = _split_large_predictions_with_sobel_y(
            image,
            predictions,
            height_ratio=_float_env("PIPE_END_YOLO_LARGE_BOX_SPLIT_HEIGHT_RATIO", DEFAULT_LARGE_BOX_SPLIT_HEIGHT_RATIO),
            max_parts=_int_env("PIPE_END_YOLO_LARGE_BOX_SPLIT_MAX_PARTS", DEFAULT_LARGE_BOX_SPLIT_MAX_PARTS),
            min_edge_strength_ratio=_float_env(
                "PIPE_END_YOLO_LARGE_BOX_SPLIT_MIN_EDGE_STRENGTH_RATIO",
                DEFAULT_LARGE_BOX_SPLIT_MIN_EDGE_STRENGTH_RATIO,
            ),
            close_x_ratio=_float_env(
                "PIPE_END_YOLO_LARGE_BOX_SPLIT_CLOSE_X_RATIO",
                DEFAULT_LARGE_BOX_SPLIT_CLOSE_X_RATIO,
            ),
            min_close_x_px=_float_env(
                "PIPE_END_YOLO_LARGE_BOX_SPLIT_MIN_CLOSE_X_PX",
                DEFAULT_LARGE_BOX_SPLIT_MIN_CLOSE_X_PX,
            ),
        )
        if large_box_split_meta.get("added_count"):
            predictions, split_postprocess = _postprocess_overlapping_predictions(
                predictions,
                overlap_threshold=overlap_threshold,
                containment_ratio=containment_ratio,
                contained_children=contained_children,
                vertical_duplicate_height_ratio=vertical_duplicate_height_ratio,
                vertical_duplicate_overlap=vertical_duplicate_overlap,
                vertical_duplicate_y_overlap=vertical_duplicate_y_overlap,
            )
            large_box_split_meta["postprocess"] = split_postprocess
    predictions = _refine_predictions_with_sobel_x(image, predictions, edge_mode=edge_mode)
    predictions.sort(key=lambda pred: pred.y_center)
    postprocess.update(
        {
            "final_count": len(predictions),
            "edge_refinement": "sobel_x",
            "edge_mode": edge_mode,
            "isolated_filter": isolated_filter_meta,
            "far_x_conf_filter": far_x_conf_filter_meta,
            "gap_recovery": gap_recovery_meta,
            "edge_gap_recovery": edge_gap_recovery_meta,
            "large_box_split": large_box_split_meta,
        }
    )
    overlay_path = output / f"{image.stem}_pipe_end_yolo_overlay.jpg"
    predictions_path = output / f"{image.stem}_pipe_end_predictions.json"
    _draw_overlay(image, predictions, overlay_path)

    x_start_list = predictions_to_x_start_list(predictions)
    payload = {
        "version": 1,
        "source": "yolo_pipe_end",
        "generated_at": _iso_now(),
        "image_path": str(image),
        "image_name": image.name,
        "model_path": str(model),
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "device": device,
        "recovery_bounds_y": None if recovery_bounds_y is None else [float(recovery_bounds_y[0]), float(recovery_bounds_y[1])],
        "spacing_stats": spacing_stats,
        "raw_prediction_count": int(raw_prediction_count),
        "postprocess": postprocess,
        "image_size": {"width": image_width, "height": image_height},
        "prediction_count": len(predictions),
        "predictions": [prediction.to_json() for prediction in predictions],
        "x_start_list": x_start_list,
        "overlay_path": str(overlay_path),
    }
    predictions_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return PipeEndInferenceResult(
        image_path=image,
        model_path=model,
        output_dir=output,
        predictions_path=predictions_path,
        overlay_path=overlay_path,
        image_width=image_width,
        image_height=image_height,
        predictions=predictions,
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        device=device,
        raw_prediction_count=int(raw_prediction_count),
        postprocess=postprocess,
        recovery_bounds_y=recovery_bounds_y,
        spacing_stats=spacing_stats,
    )
