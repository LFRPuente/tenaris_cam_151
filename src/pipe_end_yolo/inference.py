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

    @property
    def count(self) -> int:
        return len(self.predictions)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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


def _postprocess_overlapping_predictions(
    predictions: list[PipeEndPrediction],
    *,
    overlap_threshold: float = DEFAULT_OVERLAP_SUPPRESSION,
    containment_ratio: float = DEFAULT_CONTAINMENT_RATIO,
    contained_children: int = DEFAULT_CONTAINED_CHILDREN,
    vertical_duplicate_height_ratio: float = DEFAULT_VERTICAL_DUPLICATE_HEIGHT_RATIO,
    vertical_duplicate_overlap: float = DEFAULT_VERTICAL_DUPLICATE_OVERLAP,
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
    if vertical_duplicate_height_ratio > 0:
        deduped: list[PipeEndPrediction] = []
        for pred in keep:
            if deduped:
                previous = deduped[-1]
                vertical_gap = abs(float(pred.y_center) - float(previous.y_center))
                duplicate_gap = float(vertical_duplicate_height_ratio) * max(float(pred.height), float(previous.height))
                horizontal_gap = abs(float(pred.x_center) - float(previous.x_center))
                horizontal_limit = 1.25 * max(float(pred.width), float(previous.width))
                duplicate_overlap = _overlap_over_smaller(pred, previous)
                is_duplicate = (
                    vertical_gap < duplicate_gap
                    or duplicate_overlap >= float(vertical_duplicate_overlap)
                )
                if is_duplicate and horizontal_gap <= horizontal_limit:
                    vertical_suppressed += 1
                    if float(pred.confidence) > float(previous.confidence):
                        deduped[-1] = _copy_prediction(
                            pred,
                            postprocess_flags=tuple([*pred.postprocess_flags, "kept_vertical_duplicate"]),
                        )
                    else:
                        deduped[-1] = _copy_prediction(
                            previous,
                            postprocess_flags=tuple([*previous.postprocess_flags, "kept_vertical_duplicate"]),
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
        "final_count_before_refine": len(keep),
        "overlap_threshold": float(overlap_threshold),
        "vertical_duplicate_height_ratio": float(vertical_duplicate_height_ratio),
        "vertical_duplicate_overlap": float(vertical_duplicate_overlap),
        "containment_ratio": float(containment_ratio),
        "contained_children": int(contained_children),
    }


def _smooth_profile(profile: np.ndarray) -> np.ndarray:
    if profile.size < 5:
        return profile.astype(np.float32, copy=False)
    kernel = min(9, int(profile.size) if int(profile.size) % 2 == 1 else int(profile.size) - 1)
    kernel = max(3, kernel)
    return cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32), (kernel, 1), 0).reshape(-1)


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
    overlap_threshold = float(os.environ.get("PIPE_END_YOLO_OVERLAP_THRESHOLD", DEFAULT_OVERLAP_SUPPRESSION))
    containment_ratio = float(os.environ.get("PIPE_END_YOLO_CONTAINMENT_RATIO", DEFAULT_CONTAINMENT_RATIO))
    contained_children = int(os.environ.get("PIPE_END_YOLO_CONTAINED_CHILDREN", DEFAULT_CONTAINED_CHILDREN))
    vertical_duplicate_height_ratio = float(
        os.environ.get("PIPE_END_YOLO_VERTICAL_DUPLICATE_HEIGHT_RATIO", DEFAULT_VERTICAL_DUPLICATE_HEIGHT_RATIO)
    )
    vertical_duplicate_overlap = float(
        os.environ.get("PIPE_END_YOLO_VERTICAL_DUPLICATE_OVERLAP", DEFAULT_VERTICAL_DUPLICATE_OVERLAP)
    )
    edge_mode = str(os.environ.get("PIPE_END_YOLO_EDGE_MODE", DEFAULT_EDGE_MODE)).strip().lower() or DEFAULT_EDGE_MODE
    predictions, postprocess = _postprocess_overlapping_predictions(
        predictions,
        overlap_threshold=overlap_threshold,
        containment_ratio=containment_ratio,
        contained_children=contained_children,
        vertical_duplicate_height_ratio=vertical_duplicate_height_ratio,
        vertical_duplicate_overlap=vertical_duplicate_overlap,
    )
    predictions = _refine_predictions_with_sobel_x(image, predictions, edge_mode=edge_mode)
    predictions.sort(key=lambda pred: pred.y_center)
    postprocess.update(
        {
            "final_count": len(predictions),
            "edge_refinement": "sobel_x",
            "edge_mode": edge_mode,
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
    )
