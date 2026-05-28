from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

import cv2

from src.cam151_ref_detection.roi_store import load_rois
from src.cam151_ref_detection.tube_detection_preview import TubeDetectionPreviewResult
from src.cam151_ref_detection.tube_matcher_proc import export_tube_measurements
from src.pipe_end_yolo import PipeEndInferenceResult, predictions_to_x_start_list, run_pipe_end_inference
from src.pipe_end_yolo.annotation_context import model_path_for_side, spacing_stats_for_side

from .config import CameraPipelineConfig, PipelineOutputConfig, repo_root
from .notebook_style_detection import detect_tubes_like_notebook


@dataclass(frozen=True)
class CameraProcessingResult:
    side: str
    image_path: Path
    roi_path: Path
    output_dir: Path
    measurement_export_path: Path
    tube_count: int
    detection_result: TubeDetectionPreviewResult
    tube_measurements: list[dict[str, Any]]
    pipe_end_yolo: PipeEndInferenceResult | None = None


def _line_x_at_y(top_point: list[float], bottom_point: list[float], y_value: float) -> float:
    x0, y0 = float(top_point[0]), float(top_point[1])
    x1, y1 = float(bottom_point[0]), float(bottom_point[1])
    if abs(y1 - y0) < 1e-8:
        return float(0.5 * (x0 + x1))
    t = (float(y_value) - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def _select_reference_line(result: TubeDetectionPreviewResult) -> dict[str, Any] | None:
    preferred = "ref_01" if result.processing_mode == "cam152" else "ref_02"
    for ref in result.reference_lines:
        if str(ref.get("label") or "") == preferred:
            return ref
    if result.reference_lines:
        return result.reference_lines[-1]
    return None


def _scale_samples_for_export(result: TubeDetectionPreviewResult) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample in result.scale_samples or []:
        p1 = sample.get("p1_warp") or sample.get("p1")
        p2 = sample.get("p2_warp") or sample.get("p2")
        samples.append(
            {
                "label": str(sample.get("label") or ""),
                "distance_in": None if sample.get("distance_in") is None else float(sample["distance_in"]),
                "distance_px": None if sample.get("distance_px") is None else float(sample["distance_px"]),
                "px_per_in": None if sample.get("px_per_in") is None else float(sample["px_per_in"]),
                "p1_warp": [float(p1[0]), float(p1[1])] if p1 else None,
                "p2_warp": [float(p2[0]), float(p2[1])] if p2 else None,
            }
        )
    return samples


def _reference_line_for_export(result: TubeDetectionPreviewResult) -> dict[str, list[float]] | None:
    ref = _select_reference_line(result)
    if ref is None:
        return None
    top = ref.get("top_point")
    bottom = ref.get("bottom_point")
    if not top or not bottom:
        return None
    return {
        "mark_02": [float(top[0]), float(top[1])],
        "mark_03": [float(bottom[0]), float(bottom[1])],
    }


def _pipe_end_yolo_enabled() -> bool:
    raw = str(os.environ.get("PIPE_END_YOLO_ENABLED", "1")).strip().lower()
    return raw in {"1", "true", "yes", "on", "yolo", "pipe_end"}


def _pipe_end_yolo_device() -> str | None:
    raw = str(os.environ.get("PIPE_END_YOLO_DEVICE", "")).strip()
    return raw or None


def _float_env(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _pipe_end_yolo_conf(result: TubeDetectionPreviewResult) -> float:
    camera_key = "CAM152" if result.processing_mode == "cam152" else "CAM151"
    camera_specific = f"PIPE_END_YOLO_CONF_{camera_key}"
    if str(os.environ.get(camera_specific, "")).strip():
        return _float_env(camera_specific, 0.20)
    if str(os.environ.get("PIPE_END_YOLO_CONF", "")).strip():
        return _float_env("PIPE_END_YOLO_CONF", 0.20)
    return _float_env("PIPE_END_ANNOTATOR_PREDICT_CONF", 0.20)


def _pipe_end_yolo_iou() -> float:
    return _float_env("PIPE_END_YOLO_IOU", 0.50)


def _sam_payload_y_bounds(payload: dict[str, Any]) -> tuple[float, float] | None:
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


def _canonical_boundary_stems(image_path: Path, side: str) -> list[str]:
    stems = {image_path.stem}
    side_key = str(side).strip()
    if side_key in {"151", "152"}:
        stems.add(image_path.stem.replace(f"cam_{side_key}_", f"cam{side_key}_"))
        stems.add(image_path.stem.replace(f"cam{side_key}_", f"cam_{side_key}_"))
    return sorted(stem for stem in stems if stem)


def _image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def _latest_sam_boundary_meta(
    source_image_path: Path,
    side: str,
    *,
    target_image_path: Path | None = None,
) -> dict[str, Any] | None:
    side_key = str(side).strip()
    camera_dir = f"cam{side_key}"
    if side_key not in {"151", "152"}:
        return None

    target_size = _image_size(target_image_path) if target_image_path is not None else None
    boundaries_root = repo_root() / "sam_boundary_detection" / "sam2p1_boundary_app" / "boundaries"
    candidates: list[dict[str, Any]] = []
    for source in ("sam_full_image", "pipe_end_span"):
        for stem in _canonical_boundary_stems(source_image_path, side_key):
            path = boundaries_root / source / camera_dir / f"{stem}.json"
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            bounds = _sam_payload_y_bounds(payload)
            if bounds is None:
                continue
            boundary_width = payload.get("width")
            boundary_height = payload.get("height")
            if target_size is not None and boundary_width is not None and boundary_height is not None:
                if abs(int(boundary_width) - target_size[0]) > 2 or abs(int(boundary_height) - target_size[1]) > 2:
                    continue
            top, bottom = bounds
            candidates.append(
                {
                    "source": source,
                    "path": str(path.relative_to(repo_root())),
                    "bounds_y": [float(top), float(bottom)],
                    "height_px": float(bottom - top),
                    "mask_area_px": payload.get("mask_area_px"),
                    "sam_model": payload.get("sam_model"),
                    "mtime": path.stat().st_mtime,
                }
            )
    if not candidates:
        return None
    selected = max(candidates, key=lambda item: (float(item["height_px"]), float(item["mtime"])))
    selected.pop("mtime", None)
    return selected


def _sam_boundary_autogen_enabled() -> bool:
    raw = str(os.environ.get("SAM_BOUNDARY_AUTOGEN_ENABLED", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _sam_boundary_model_name() -> str:
    return str(os.environ.get("SAM_BOUNDARY_MODEL", "sam2.1_s.pt")).strip() or "sam2.1_s.pt"


def _sam_boundary_device() -> str:
    raw = str(os.environ.get("SAM_BOUNDARY_DEVICE", "")).strip()
    return raw or _pipe_end_yolo_device() or "0"


def _ensure_sam_boundary_from_warp(
    source_image_path: Path,
    side: str,
    warp_image_path: Path,
) -> dict[str, Any] | None:
    if not _sam_boundary_autogen_enabled():
        return None
    side_key = str(side).strip()
    if side_key not in {"151", "152"}:
        return None
    if not warp_image_path.exists():
        return None

    try:
        from apps.pipe_end_annotator.annotate_app import AppPaths, run_sam_boundary
    except Exception as exc:
        print(f"[sam_boundary] annotator flow unavailable: {exc}")
        return None

    root = repo_root() / "pipe_end_detection"
    images_root = repo_root() / "sam_boundary_detection" / "sam2p1_boundary_app" / "mvp_warp_images"
    rel = Path(f"cam{side_key}") / f"{source_image_path.stem}{warp_image_path.suffix.lower() or '.jpg'}"
    image_path = images_root / rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if (not image_path.exists()) or image_path.stat().st_mtime < warp_image_path.stat().st_mtime:
            shutil.copy2(warp_image_path, image_path)
    except Exception as exc:
        print(f"[sam_boundary] could not stage warp image: {exc}")
        return None

    paths = AppPaths(
        root=root,
        images_root=images_root,
        labels_root=root / "annotation_pool" / "labels",
        predictions_root=root / "predictions" / "sam_boundary" / "labels",
        status_path=root / "annotation_pool" / "image_status.json",
        auto_train=False,
        min_train_images=4,
        train_epochs=40,
        train_imgsz=1280,
        train_batch=1,
        base_model="yolo11n.pt",
        train_device=_sam_boundary_device(),
        roi_toml_151=None,
        roi_toml_152=None,
        raw_image_151=None,
        raw_image_152=None,
    )
    try:
        payload = run_sam_boundary(
            paths,
            rel,
            side="right",
            sam_model=_sam_boundary_model_name(),
            prompt_mode="pipe_end_span",
        )
    except Exception as exc:
        print(f"[sam_boundary] generation failed for {source_image_path.name}: {exc}")
        return None
    return payload if isinstance(payload, dict) else None


def _sam_boundary_meta_from_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    bounds = _sam_payload_y_bounds(payload)
    if bounds is None:
        return None
    top, bottom = bounds
    return {
        "source": str(payload.get("prompt_source") or "pipe_end_span"),
        "path": str(payload.get("boundary_path") or ""),
        "bounds_y": [float(top), float(bottom)],
        "height_px": float(bottom - top),
        "mask_area_px": payload.get("mask_area_px"),
        "sam_model": payload.get("sam_model"),
        "prompt_pipe_end_count": payload.get("prompt_pipe_end_count"),
        "prompt_pipe_end_raw_count": payload.get("prompt_pipe_end_raw_count"),
        "prompt_box": payload.get("prompt_box"),
    }


def _apply_pipe_end_yolo_detection(
    result: TubeDetectionPreviewResult,
    output_dir: Path,
    *,
    side: str,
    recovery_bounds_y: tuple[float, float] | None = None,
    spacing_stats: dict[str, Any] | None = None,
) -> PipeEndInferenceResult:
    yolo_output_dir = output_dir / "pipe_end_yolo"
    yolo_result = run_pipe_end_inference(
        result.homography.warp_path,
        yolo_output_dir,
        model_path=model_path_for_side(side),
        conf=_pipe_end_yolo_conf(result),
        iou=_pipe_end_yolo_iou(),
        device=_pipe_end_yolo_device(),
        recovery_bounds_y=recovery_bounds_y,
        spacing_stats=spacing_stats,
    )
    x_start_list = predictions_to_x_start_list(yolo_result.predictions)
    if not x_start_list:
        raise ValueError(
            f"YOLO pipe_end no detecto tubos en {result.homography.warp_path}. "
            "Revisa el modelo o desactiva PIPE_END_YOLO_ENABLED=0 para usar el detector clasico."
        )

    result.x_start_list = x_start_list
    result.tube_count = len(x_start_list)
    result.detection_overlay_path = yolo_result.overlay_path
    result.processing_stage = "yolo_pipe_end"
    result.dominant_period = None
    result.energy_start_index = int(min(float(item["y_center"]) for item in x_start_list))
    result.peaks_index = [int(round(float(item["y_center"]))) for item in x_start_list]
    result.peaks_index_dom = list(result.peaks_index)
    result.rejected_tube_gaps = []
    return yolo_result


def build_tube_measurements(result: TubeDetectionPreviewResult) -> list[dict[str, Any]]:
    if result.px_per_in is None or result.px_per_in <= 1e-6:
        raise ValueError("No hay escala px/in valida para exportar mediciones.")
    ref_line = _select_reference_line(result)
    if ref_line is None:
        raise ValueError("No hay linea de referencia valida para exportar mediciones.")

    measurements: list[dict[str, Any]] = []
    for item in result.x_start_list:
        tube_idx = int(item.get("tube_idx") or len(measurements) + 1)
        x_start = float(item.get("x_start", item.get("x_local", 0.0)))
        x_smooth = item.get("x_end_estimate", item.get("x_seed"))
        y_center = float(item.get("y_center", 0.0))
        ref_x = _line_x_at_y(ref_line["top_point"], ref_line["bottom_point"], y_center)
        offset_px = float(x_start - ref_x)
        distance_in = abs(offset_px) / float(result.px_per_in)
        measurement: dict[str, Any] = {
            "tube_idx": tube_idx,
            "x_start_warp": x_start,
            "x_start_raw_warp": x_start,
            "x_start_smooth_warp": None if x_smooth is None else float(x_smooth),
            "y_center_warp": y_center,
            "ref_x": ref_x,
            "offset_px": offset_px,
            "distance_in": distance_in,
        }
        for passthrough_key in (
            "confidence",
            "source",
            "box_xyxy",
            "box_xywh",
            "yolo_box_center",
            "refined_pipe_end",
        ):
            if passthrough_key in item:
                measurement[passthrough_key] = item[passthrough_key]
        if result.processing_mode == "cam152":
            # cam152 is mirrored in the backend for cam_152_* images, so a
            # negative warp offset represents the far/right-side extension.
            measurement["relative_position"] = "after" if offset_px < 0 else "before"
        measurements.append(measurement)

    measurements.sort(key=lambda row: int(row["tube_idx"]))
    return measurements


def _write_analysis_image(result: TubeDetectionPreviewResult, output_dir: Path) -> Path | None:
    warp_path = result.homography.warp_path
    image = cv2.imread(str(warp_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    app_dir = output_dir / "manual_measure_app"
    app_dir.mkdir(parents=True, exist_ok=True)
    image_path = app_dir / "analysis_warp_current.jpg"
    cv2.imwrite(str(image_path), image)
    return image_path


def process_camera(config: CameraPipelineConfig, outputs: PipelineOutputConfig) -> CameraProcessingResult:
    roi_payload = load_rois(config.roi_path)
    output_dir = outputs.artifact_root / f"cam{config.side}" / config.image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    result = detect_tubes_like_notebook(
        image_path=config.image_path,
        roi_payload=roi_payload,
        output_dir=output_dir,
    )
    generated_sam_boundary = _ensure_sam_boundary_from_warp(config.image_path, config.side, result.homography.warp_path)
    sam_boundary = _sam_boundary_meta_from_payload(generated_sam_boundary)
    if sam_boundary is None and not _sam_boundary_autogen_enabled():
        sam_boundary = _latest_sam_boundary_meta(
            config.image_path,
            config.side,
            target_image_path=result.homography.warp_path,
        )
    recovery_bounds_y = None
    if sam_boundary is not None:
        bounds_y = sam_boundary.get("bounds_y")
        if isinstance(bounds_y, list) and len(bounds_y) >= 2:
            recovery_bounds_y = (float(bounds_y[0]), float(bounds_y[1]))
    spacing_stats = spacing_stats_for_side(config.side)

    yolo_result: PipeEndInferenceResult | None = None
    detection_source = "notebook_style_sobel_x_roi"
    if _pipe_end_yolo_enabled():
        yolo_result = _apply_pipe_end_yolo_detection(
            result,
            output_dir,
            side=config.side,
            recovery_bounds_y=recovery_bounds_y,
            spacing_stats=spacing_stats,
        )
        detection_source = "yolo_pipe_end"

    tube_measurements = build_tube_measurements(result)
    analysis_image_path = _write_analysis_image(result, output_dir)

    extra_meta = {
        "px_per_in_nb": None if result.px_per_in is None else float(result.px_per_in),
        "scale_samples": _scale_samples_for_export(result),
        "detection_roi": None if result.detection_roi is None else [float(v) for v in result.detection_roi],
        "reference_line_warp": _reference_line_for_export(result),
        "analysis_image_path": None if analysis_image_path is None else str(analysis_image_path),
        "processing_mode": result.processing_mode,
        "processing_stage": result.processing_stage,
        "detection_source": detection_source,
        "backend_pipeline": "tenaris_tube_pipeline",
    }
    if sam_boundary is not None:
        extra_meta["sam_boundary"] = sam_boundary
        extra_meta["sam_boundary_bounds_y"] = sam_boundary["bounds_y"]
    if yolo_result is not None:
        extra_meta["pipe_end_yolo"] = {
            "enabled": True,
            "model_path": str(yolo_result.model_path),
            "raw_prediction_count": int(yolo_result.raw_prediction_count),
            "prediction_count": int(yolo_result.count),
            "postprocess": dict(yolo_result.postprocess or {}),
            "predictions_path": str(yolo_result.predictions_path),
            "overlay_path": str(yolo_result.overlay_path),
            "imgsz": int(yolo_result.imgsz),
            "conf": float(yolo_result.conf),
            "iou": float(yolo_result.iou),
            "device": yolo_result.device,
            "recovery_bounds_y": (
                None
                if yolo_result.recovery_bounds_y is None
                else [float(yolo_result.recovery_bounds_y[0]), float(yolo_result.recovery_bounds_y[1])]
            ),
            "spacing_stats": yolo_result.spacing_stats,
        }

    measurement_export_path = export_tube_measurements(
        config.side,
        tube_measurements,
        image_path=str(config.image_path),
        roi_path=str(config.roi_path),
        source_notebook=config.source_name,
        dataset_name=config.dataset_name,
        extra_meta=extra_meta,
        output_dir=outputs.matcher_input_dir,
    )

    return CameraProcessingResult(
        side=config.side,
        image_path=config.image_path,
        roi_path=config.roi_path,
        output_dir=output_dir,
        measurement_export_path=measurement_export_path,
        tube_count=len(tube_measurements),
        detection_result=result,
        tube_measurements=tube_measurements,
        pipe_end_yolo=yolo_result,
    )
