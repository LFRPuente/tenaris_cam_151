from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

import cv2


PIPE_END_CLASS_ID = 0
PIPE_END_CLASS_NAME = "pipe_end"
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.20
DEFAULT_IOU = 0.50


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

    def to_json(self) -> dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "box_xyxy": [float(self.x1), float(self.y1), float(self.x2), float(self.y2)],
            "box_xywh": [float(self.x_center), float(self.y_center), float(self.width), float(self.height)],
        }


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


def _draw_overlay(image_path: Path, predictions: list[PipeEndPrediction], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"No se pudo leer imagen para overlay YOLO: {image_path}")

    ordered = sorted(predictions, key=lambda pred: pred.y_center)
    total = len(ordered)
    for pos, pred in enumerate(ordered):
        tube_idx = max(1, total - pos)
        p1 = (int(round(pred.x1)), int(round(pred.y1)))
        p2 = (int(round(pred.x2)), int(round(pred.y2)))
        center = (int(round(pred.x_center)), int(round(pred.y_center)))
        cv2.rectangle(image, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(image, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
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

    The historical pipeline numbers tubes bottom-to-top: the lowest visible pipe is tube 1.
    YOLO detections are sorted top-to-bottom and assigned the same numbering convention.
    """
    ordered = sorted(predictions, key=lambda pred: pred.y_center)
    total = len(ordered)
    x_start_list: list[dict[str, Any]] = []
    for pos, pred in enumerate(ordered):
        tube_idx = int(max(1, total - pos))
        x_start = float(pred.x_center)
        y_center = float(pred.y_center)
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

    predictions.sort(key=lambda pred: pred.y_center)
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
    )
