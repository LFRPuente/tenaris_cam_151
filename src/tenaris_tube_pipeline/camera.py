from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from src.cam151_ref_detection.roi_store import load_rois
from src.cam151_ref_detection.tube_detection_preview import TubeDetectionPreviewResult
from src.cam151_ref_detection.tube_matcher_proc import export_tube_measurements

from .config import CameraPipelineConfig, PipelineOutputConfig
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
        if result.processing_mode == "cam152":
            measurement["relative_position"] = "before" if offset_px < 0 else "after"
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
    tube_measurements = build_tube_measurements(result)
    analysis_image_path = _write_analysis_image(result, output_dir)

    measurement_export_path = export_tube_measurements(
        config.side,
        tube_measurements,
        image_path=str(config.image_path),
        roi_path=str(config.roi_path),
        source_notebook=config.source_name,
        dataset_name=config.dataset_name,
        extra_meta={
            "px_per_in_nb": None if result.px_per_in is None else float(result.px_per_in),
            "scale_samples": _scale_samples_for_export(result),
            "detection_roi": None if result.detection_roi is None else [float(v) for v in result.detection_roi],
            "reference_line_warp": _reference_line_for_export(result),
            "analysis_image_path": None if analysis_image_path is None else str(analysis_image_path),
            "processing_mode": result.processing_mode,
            "processing_stage": result.processing_stage,
            "backend_pipeline": "tenaris_tube_pipeline",
        },
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
    )
