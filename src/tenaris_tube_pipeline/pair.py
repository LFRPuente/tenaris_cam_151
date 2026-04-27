from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.cam151_ref_detection.tube_matcher_proc import _load_dataset, build_result_payload, write_match_results

from .camera import CameraProcessingResult, process_camera
from .config import CameraPipelineConfig, PipelineOutputConfig


@dataclass(frozen=True)
class TubePairProcessingResult:
    cam151: CameraProcessingResult
    cam152: CameraProcessingResult
    match_payload: dict[str, Any]
    result_json_path: Path
    result_xlsx_path: Path
    latest_json_path: Path
    latest_xlsx_path: Path


def process_tube_pair(
    cam151: CameraPipelineConfig,
    cam152: CameraPipelineConfig,
    outputs: PipelineOutputConfig,
    *,
    output_stem: str | None = None,
) -> TubePairProcessingResult:
    cam151_result = process_camera(cam151, outputs)
    cam152_result = process_camera(cam152, outputs)

    left_dataset = _load_dataset(cam151_result.measurement_export_path, expected_side="151")
    right_dataset = _load_dataset(cam152_result.measurement_export_path, expected_side="152")
    match_payload = build_result_payload(left_dataset, right_dataset, left_dataset["items"], right_dataset["items"])
    json_path, xlsx_path, latest_json_path, latest_xlsx_path = write_match_results(
        match_payload,
        output_dir=outputs.match_output_dir,
        stem=output_stem,
    )

    return TubePairProcessingResult(
        cam151=cam151_result,
        cam152=cam152_result,
        match_payload=match_payload,
        result_json_path=json_path,
        result_xlsx_path=xlsx_path,
        latest_json_path=latest_json_path,
        latest_xlsx_path=latest_xlsx_path,
    )
