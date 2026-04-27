"""Manual tube measurement app over clean warp analysis images."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse

import cv2

from .tube_matcher_proc import (
    BASE_CENTRAL_IN,
    LEFT_REFERENCE_IN,
    RIGHT_REFERENCE_IN,
    _display_jupyter_link,
    _extract_notebook_png_by_hints,
    _load_artifact_warp_image,
    _load_dataset,
    _png_dimensions,
    _repo_root,
    _write_simple_xlsx,
    find_latest_measurement_export,
)


MANUAL_RESULT_VERSION = 1


@dataclass
class ManualTubeMeasureRunResult:
    confirmed: bool
    result_payload: dict[str, Any] | None
    result_json_path: Path | None
    result_xlsx_path: Path | None
    latest_json_path: Path | None
    latest_xlsx_path: Path | None


def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _default_output_dir() -> Path:
    return _repo_root() / "artifacts" / "manual_tube_measurements"


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return float(number)


def _median_or_none(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(median(clean))


def _normalize_reference_line(raw_line: Any) -> dict[str, list[float]] | None:
    if not isinstance(raw_line, dict):
        return None
    mark_02 = _float_pair(raw_line.get("mark_02"))
    mark_03 = _float_pair(raw_line.get("mark_03"))
    if mark_02 is None or mark_03 is None:
        return None
    return {
        "mark_02": [round(mark_02[0], 3), round(mark_02[1], 3)],
        "mark_03": [round(mark_03[0], 3), round(mark_03[1], 3)],
    }


def _normalize_scale_samples(raw_scales: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_scales, list):
        return normalized
    for index, raw_scale in enumerate(raw_scales, start=1):
        if not isinstance(raw_scale, dict):
            continue
        p1 = _float_pair(raw_scale.get("p1_warp") or raw_scale.get("p1"))
        p2 = _float_pair(raw_scale.get("p2_warp") or raw_scale.get("p2"))
        distance_in = _float_or_none(raw_scale.get("distance_in"))
        distance_px = _float_or_none(raw_scale.get("distance_px"))
        px_per_in = _float_or_none(raw_scale.get("px_per_in"))
        if p1 is None or p2 is None or distance_in in {None, 0.0}:
            continue
        if distance_px is None and px_per_in is not None:
            distance_px = float(px_per_in * distance_in)
        if px_per_in is None and distance_px is not None and distance_in not in {None, 0.0}:
            px_per_in = float(distance_px / distance_in)
        normalized.append(
            {
                "label": str(raw_scale.get("label") or f"wscale_{index:02d}"),
                "distance_in": float(distance_in),
                "distance_px": distance_px,
                "px_per_in": px_per_in,
                "p1": [round(p1[0], 3), round(p1[1], 3)],
                "p2": [round(p2[0], 3), round(p2[1], 3)],
            }
        )
    return normalized


def _float_pair(value: Any) -> tuple[float, float] | None:
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (x == x and y == y):
        return None
    return float(x), float(y)


def _resolve_dataset_image_path(dataset: dict[str, Any]) -> Path | None:
    extra_meta = dataset.get("extra_meta") or {}
    if not isinstance(extra_meta, dict):
        extra_meta = {}
    raw_path = str(extra_meta.get("analysis_image_path") or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = _repo_root() / path
    return path if path.exists() else None


def _extract_reference_line_from_notebook(source_notebook: Any) -> dict[str, list[float]] | None:
    notebook_path = Path(source_notebook) if source_notebook else None
    if notebook_path is None:
        return None
    if not notebook_path.is_absolute():
        notebook_path = (_repo_root() / "notebooks" / notebook_path.name) if not str(notebook_path).startswith("notebooks") else (_repo_root() / notebook_path)
    if not notebook_path.exists():
        return None
    try:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    mark02: tuple[float, float] | None = None
    mark03: tuple[float, float] | None = None
    pattern = re.compile(
        r"mark_(0[23]).*?warp=\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)",
        re.IGNORECASE,
    )
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict):
            continue
        for output in cell.get("outputs", []):
            if not isinstance(output, dict):
                continue
            text_chunks = []
            if isinstance(output.get("text"), list):
                text_chunks.extend(str(part) for part in output.get("text", []))
            elif output.get("text"):
                text_chunks.append(str(output.get("text")))
            data = output.get("data")
            if isinstance(data, dict):
                plain = data.get("text/plain")
                if isinstance(plain, list):
                    text_chunks.extend(str(part) for part in plain)
                elif plain:
                    text_chunks.append(str(plain))
            output_text = "".join(text_chunks)
            if not output_text:
                continue
            for label, x_text, y_text in pattern.findall(output_text):
                point = (float(x_text), float(y_text))
                if label == "02":
                    mark02 = point
                elif label == "03":
                    mark03 = point
    if mark02 is None or mark03 is None:
        return None
    return {
        "mark_02": [round(mark02[0], 3), round(mark02[1], 3)],
        "mark_03": [round(mark03[0], 3), round(mark03[1], 3)],
    }


def _image_state_for_dataset(side: str, dataset: dict[str, Any], asset_url: str) -> dict[str, Any]:
    asset_bytes: bytes | None = None
    asset_type = "image/jpeg"
    width: int | None = None
    height: int | None = None

    exported_image_path = _resolve_dataset_image_path(dataset)
    if exported_image_path is not None:
        image = cv2.imread(str(exported_image_path), cv2.IMREAD_COLOR)
        if image is not None and image.size > 0:
            height, width = image.shape[:2]
            asset_bytes = exported_image_path.read_bytes()
            suffix = exported_image_path.suffix.lower()
            asset_type = "image/png" if suffix == ".png" else "image/jpeg"

    if asset_bytes is None:
        artifact = _load_artifact_warp_image(side, dataset)
        if artifact is not None:
            asset_bytes, asset_type, width, height = artifact
        else:
            png_bytes = _extract_notebook_png_by_hints(
                dataset.get("source_notebook"),
                ["Crop de la detection ROI", "Warp recortado para analisis", "Homografia: warp rectificado"],
            )
            width, height = _png_dimensions(png_bytes)
            asset_bytes = png_bytes
            asset_type = "image/png"

    if not asset_bytes or width is None or height is None:
        return {
            "available": False,
            "url": None,
            "width": None,
            "height": None,
            "asset_bytes": None,
            "asset_content_type": None,
        }

    return {
        "available": True,
        "url": asset_url,
        "width": int(width),
        "height": int(height),
        "asset_bytes": asset_bytes,
        "asset_content_type": asset_type,
    }


def _algorithm_state_for_dataset(side: str, dataset: dict[str, Any]) -> dict[str, Any]:
    items = dataset.get("items") or []
    ref_x_values: list[float] = []
    px_per_in_values: list[float] = []
    algorithm_by_tube: dict[str, dict[str, Any]] = {}
    extra_meta = dataset.get("extra_meta") or {}
    if not isinstance(extra_meta, dict):
        extra_meta = {}
    ref_line = _normalize_reference_line(extra_meta.get("reference_line_warp")) or _extract_reference_line_from_notebook(dataset.get("source_notebook"))
    px_per_in_meta = _float_or_none(extra_meta.get("px_per_in_nb"))
    scale_samples = _normalize_scale_samples(extra_meta.get("scale_samples"))

    for item in items:
        if not isinstance(item, dict):
            continue
        tube_idx = int(item.get("tube_idx") or 0)
        source_measurement = item.get("source_measurement")
        if not isinstance(source_measurement, dict):
            source_measurement = {}

        ref_x = _float_or_none(source_measurement.get("ref_x"))
        offset_px = _float_or_none(source_measurement.get("offset_px"))
        distance_in = _float_or_none(item.get("distance_in"))
        if ref_x is not None:
            ref_x_values.append(ref_x)
        if offset_px is not None and distance_in not in {None, 0.0}:
            px_per_in_values.append(abs(float(offset_px)) / float(distance_in))

        if side == "151":
            r_value = float(distance_in - LEFT_REFERENCE_IN) if distance_in is not None else None
            relative_position = None
        else:
            relative_position = str(item.get("relative_position") or "after").strip().lower()
            if relative_position == "before":
                r_value = float(RIGHT_REFERENCE_IN - distance_in) if distance_in is not None else None
            else:
                relative_position = "after"
                r_value = float(RIGHT_REFERENCE_IN + distance_in) if distance_in is not None else None

        algorithm_by_tube[str(tube_idx)] = {
            "tube_idx": tube_idx,
            "distance_in": distance_in,
            "relative_position": relative_position,
            "r_value": r_value,
            "x": _float_or_none(
                source_measurement.get("x_start_raw_warp")
                or source_measurement.get("x_start_warp")
                or source_measurement.get("x_start_smooth_warp")
            ),
            "y": _float_or_none(source_measurement.get("y_center_warp")),
            "ref_x": ref_x,
            "offset_px": offset_px,
        }

    return {
        "algorithm_by_tube": algorithm_by_tube,
        "ref_x": _median_or_none(ref_x_values),
        "px_per_in": px_per_in_meta if px_per_in_meta is not None else _median_or_none(px_per_in_values),
        "tube_count": len(algorithm_by_tube),
        "ref_line": ref_line,
        "scale_samples": scale_samples,
    }


def _initial_state(left_dataset: dict[str, Any], right_dataset: dict[str, Any], image_state: dict[str, Any]) -> dict[str, Any]:
    left_algo = _algorithm_state_for_dataset("151", left_dataset)
    right_algo = _algorithm_state_for_dataset("152", right_dataset)
    return {
        "generated_at": _iso_now(),
        "config": {
            "base_central_in": BASE_CENTRAL_IN,
            "left_reference_in": LEFT_REFERENCE_IN,
            "right_reference_in": RIGHT_REFERENCE_IN,
        },
        "sides": {
            "151": {
                "dataset_name": left_dataset.get("dataset_name"),
                "image_name": left_dataset.get("image_name"),
                "reference_in": left_dataset.get("reference_in"),
                "ref_x": left_algo.get("ref_x"),
                "ref_line": left_algo.get("ref_line"),
                "px_per_in": left_algo.get("px_per_in"),
                "scale_samples": left_algo.get("scale_samples"),
                "algorithm_by_tube": left_algo.get("algorithm_by_tube"),
                "tube_count": left_algo.get("tube_count"),
            },
            "152": {
                "dataset_name": right_dataset.get("dataset_name"),
                "image_name": right_dataset.get("image_name"),
                "reference_in": right_dataset.get("reference_in"),
                "ref_x": right_algo.get("ref_x"),
                "ref_line": right_algo.get("ref_line"),
                "px_per_in": right_algo.get("px_per_in"),
                "scale_samples": right_algo.get("scale_samples"),
                "algorithm_by_tube": right_algo.get("algorithm_by_tube"),
                "tube_count": right_algo.get("tube_count"),
            },
        },
        "images": image_state,
    }


def _ref_x_at_y(side_state: dict[str, Any], y_value: Any) -> float | None:
    y_number = _float_or_none(y_value)
    if y_number is None:
        return _float_or_none(side_state.get("ref_x"))
    ref_line = side_state.get("ref_line")
    if isinstance(ref_line, dict):
        mark_02 = _float_pair(ref_line.get("mark_02"))
        mark_03 = _float_pair(ref_line.get("mark_03"))
        if mark_02 is not None and mark_03 is not None:
            x0, y0 = mark_02
            x1, y1 = mark_03
            if abs(y1 - y0) <= 1e-6:
                return float(0.5 * (x0 + x1))
            t = (float(y_number) - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return _float_or_none(side_state.get("ref_x"))


def _manual_measure(side: str, point: dict[str, Any] | None, side_state: dict[str, Any]) -> float | None:
    if not point:
        return None
    x_value = _float_or_none(point.get("x"))
    ref_x = _ref_x_at_y(side_state, point.get("y"))
    px_per_in = _float_or_none(side_state.get("px_per_in"))
    if x_value is None or ref_x is None or px_per_in in {None, 0.0}:
        return None
    return float(abs(x_value - ref_x) / px_per_in)


def _manual_position_152(point: dict[str, Any] | None, side_state: dict[str, Any]) -> str | None:
    if not point:
        return None
    x_value = _float_or_none(point.get("x"))
    ref_x = _ref_x_at_y(side_state, point.get("y"))
    if x_value is None or ref_x is None:
        return None
    return "before" if x_value < ref_x else "after"


def _manual_r151(point: dict[str, Any] | None, side_state: dict[str, Any]) -> float | None:
    measure = _manual_measure("151", point, side_state)
    if measure is None:
        return None
    return float(measure - LEFT_REFERENCE_IN)


def _manual_r152(point: dict[str, Any] | None, side_state: dict[str, Any]) -> float | None:
    measure = _manual_measure("152", point, side_state)
    position = _manual_position_152(point, side_state)
    if measure is None or position is None:
        return None
    if position == "before":
        return float(RIGHT_REFERENCE_IN - measure)
    return float(RIGHT_REFERENCE_IN + measure)


def _algorithm_total_for_tube(tube_number: int, left_state: dict[str, Any], right_state: dict[str, Any]) -> float | None:
    left_entry = (left_state.get("algorithm_by_tube") or {}).get(str(tube_number))
    right_entry = (right_state.get("algorithm_by_tube") or {}).get(str(tube_number))
    left_r = _float_or_none((left_entry or {}).get("r_value"))
    right_r = _float_or_none((right_entry or {}).get("r_value"))
    if left_r is None or right_r is None:
        return None
    return float(BASE_CENTRAL_IN + left_r + right_r)


def _sanitize_point(raw_point: Any, width: int | None, height: int | None) -> dict[str, float] | None:
    if not isinstance(raw_point, dict):
        return None
    x_value = _float_or_none(raw_point.get("x"))
    y_value = _float_or_none(raw_point.get("y"))
    if x_value is None or y_value is None:
        return None
    if width is not None:
        x_value = max(0.0, min(float(width), x_value))
    if height is not None:
        y_value = max(0.0, min(float(height), y_value))
    return {"x": round(float(x_value), 3), "y": round(float(y_value), 3)}


def _sanitize_rows(raw_rows: Any, image_state: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        raise ValueError("El payload de filas manuales es invalido.")
    clean_rows: list[dict[str, Any]] = []
    left_image = image_state.get("151") or {}
    right_image = image_state.get("152") or {}
    for index, raw_row in enumerate(raw_rows, start=1):
        if not isinstance(raw_row, dict):
            raise ValueError("Cada fila manual debe ser un objeto.")
        clean_rows.append(
            {
                "tube_number": int(index),
                "point151": _sanitize_point(raw_row.get("point151"), left_image.get("width"), left_image.get("height")),
                "point152": _sanitize_point(raw_row.get("point152"), right_image.get("width"), right_image.get("height")),
            }
        )
    return clean_rows


def _result_rows(rows: list[dict[str, Any]], initial_state: dict[str, Any]) -> list[dict[str, Any]]:
    left_side = initial_state["sides"]["151"]
    right_side = initial_state["sides"]["152"]
    result: list[dict[str, Any]] = []
    for row in rows:
        tube_number = int(row["tube_number"])
        point151 = row.get("point151")
        point152 = row.get("point152")
        manual_151 = _manual_measure("151", point151, left_side)
        manual_152 = _manual_measure("152", point152, right_side)
        manual_pos_152 = _manual_position_152(point152, right_side)
        manual_r151 = _manual_r151(point151, left_side)
        manual_r152 = _manual_r152(point152, right_side)
        manual_total = None
        if manual_r151 is not None and manual_r152 is not None:
            manual_total = float(BASE_CENTRAL_IN + manual_r151 + manual_r152)

        left_algo = (left_side.get("algorithm_by_tube") or {}).get(str(tube_number)) or {}
        right_algo = (right_side.get("algorithm_by_tube") or {}).get(str(tube_number)) or {}
        algo_151 = _float_or_none(left_algo.get("distance_in"))
        algo_152 = _float_or_none(right_algo.get("distance_in"))
        algo_total = _algorithm_total_for_tube(tube_number, left_side, right_side)

        result.append(
            {
                "tube_number": tube_number,
                "status": "complete" if point151 and point152 else "partial",
                "x_151": _float_or_none((point151 or {}).get("x")),
                "y_151": _float_or_none((point151 or {}).get("y")),
                "manual_measurement_151": manual_151,
                "algorithm_measurement_151": algo_151,
                "delta_151": float(manual_151 - algo_151) if manual_151 is not None and algo_151 is not None else None,
                "manual_r151": manual_r151,
                "algorithm_r151": _float_or_none(left_algo.get("r_value")),
                "x_152": _float_or_none((point152 or {}).get("x")),
                "y_152": _float_or_none((point152 or {}).get("y")),
                "manual_position_152": manual_pos_152,
                "algorithm_position_152": right_algo.get("relative_position"),
                "manual_measurement_152": manual_152,
                "algorithm_measurement_152": algo_152,
                "delta_152": float(manual_152 - algo_152) if manual_152 is not None and algo_152 is not None else None,
                "manual_r152": manual_r152,
                "algorithm_r152": _float_or_none(right_algo.get("r_value")),
                "manual_total_length_in": manual_total,
                "algorithm_total_length_in": algo_total,
                "delta_total_length_in": float(manual_total - algo_total) if manual_total is not None and algo_total is not None else None,
            }
        )
    return result


def build_manual_measure_result_payload(
    left_dataset: dict[str, Any],
    right_dataset: dict[str, Any],
    image_state: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    initial_state = _initial_state(left_dataset, right_dataset, image_state)
    return {
        "version": MANUAL_RESULT_VERSION,
        "generated_at": _iso_now(),
        "config": initial_state["config"],
        "left_input": {
            "dataset_name": left_dataset.get("dataset_name"),
            "image_name": left_dataset.get("image_name"),
            "source_notebook": left_dataset.get("source_notebook"),
            "input_path": left_dataset.get("_input_path"),
        },
        "right_input": {
            "dataset_name": right_dataset.get("dataset_name"),
            "image_name": right_dataset.get("image_name"),
            "source_notebook": right_dataset.get("source_notebook"),
            "input_path": right_dataset.get("_input_path"),
        },
        "rows": _result_rows(rows, initial_state),
    }


def write_manual_measure_results(
    payload: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    stem: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    output_root = Path(output_dir) if output_dir else _default_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    base_stem = stem or f"manual_tube_measure_{stamp}"
    json_path = output_root / f"{base_stem}.json"
    xlsx_path = output_root / f"{base_stem}.xlsx"
    latest_json_path = output_root / "manual_tube_measure_latest.json"
    latest_xlsx_path = output_root / "manual_tube_measure_latest.xlsx"

    payload_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(payload_text, encoding="utf-8")
    latest_json_path.write_text(payload_text, encoding="utf-8")

    config_sheet = [payload.get("config") or {}]
    _write_simple_xlsx(xlsx_path, [("manual_measurements", payload.get("rows") or []), ("config", config_sheet)])
    latest_xlsx_path.write_bytes(xlsx_path.read_bytes())
    return json_path, xlsx_path, latest_json_path, latest_xlsx_path


def _manual_measure_html(left_dataset: dict[str, Any], right_dataset: dict[str, Any], image_state: dict[str, Any]) -> str:
    html_path = Path(__file__).with_name("manual_tube_measure.html")
    html_template = html_path.read_text(encoding="utf-8")
    initial_state = _initial_state(left_dataset, right_dataset, image_state)
    return html_template.replace("__INITIAL_STATE__", json.dumps(initial_state, ensure_ascii=False))


def run_manual_tube_measure_app(
    cam151_input: str | Path | dict[str, Any] | None = None,
    cam152_input: str | Path | dict[str, Any] | None = None,
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_stem: str | None = None,
    timeout: float | None = None,
    open_browser: bool = True,
    on_ready: Any | None = None,
) -> ManualTubeMeasureRunResult:
    runtime_lock = threading.Lock()

    def _current_sources() -> tuple[str | Path | dict[str, Any], str | Path | dict[str, Any]]:
        left_source = cam151_input if cam151_input is not None else find_latest_measurement_export("151", input_dir=input_dir)
        right_source = cam152_input if cam152_input is not None else find_latest_measurement_export("152", input_dir=input_dir)
        return left_source, right_source

    def _build_runtime_snapshot() -> dict[str, Any]:
        left_source, right_source = _current_sources()
        left_dataset = _load_dataset(left_source, expected_side="151")
        right_dataset = _load_dataset(right_source, expected_side="152")
        image_state = {
            "151": _image_state_for_dataset("151", left_dataset, "/asset/cam151-roi.jpg"),
            "152": _image_state_for_dataset("152", right_dataset, "/asset/cam152-roi.jpg"),
        }
        asset_map: dict[str, tuple[bytes, str]] = {}
        for image_meta in image_state.values():
            asset_bytes = image_meta.pop("asset_bytes", None)
            asset_type = image_meta.pop("asset_content_type", None)
            asset_url = image_meta.get("url")
            if asset_bytes and asset_url:
                asset_map[str(asset_url)] = (asset_bytes, str(asset_type or "application/octet-stream"))
        html = _manual_measure_html(left_dataset, right_dataset, image_state)
        return {
            "left_dataset": left_dataset,
            "right_dataset": right_dataset,
            "image_state": image_state,
            "asset_map": asset_map,
            "html_bytes": html.encode("utf-8"),
        }

    runtime_box: dict[str, Any] = _build_runtime_snapshot()
    done = threading.Event()
    result_box: dict[str, Any] = {
        "confirmed": False,
        "payload": None,
        "json_path": None,
        "xlsx_path": None,
        "latest_json_path": None,
        "latest_xlsx_path": None,
    }

    class ManualMeasureHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                with runtime_lock:
                    runtime_box.update(_build_runtime_snapshot())
                    html_bytes = runtime_box["html_bytes"]
                self._send_bytes(html_bytes, "text/html; charset=utf-8")
                return
            with runtime_lock:
                asset = runtime_box["asset_map"].get(parsed.path)
            if asset is not None:
                self._send_bytes(asset[0], asset[1])
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/result":
                self._send_json({"error": "not found"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw) if raw else {}
            except Exception as exc:
                self._send_json({"error": f"request invalido: {exc}"}, status=400)
                return

            if not isinstance(payload, dict):
                self._send_json({"error": "payload invalido"}, status=400)
                return

            action = str(payload.get("action") or "").strip().lower()
            if action == "cancel":
                result_box["confirmed"] = False
                done.set()
                self._send_json({"ok": True, "confirmed": False, "closed": True})
                return

            if not payload.get("confirmed"):
                self._send_json({"ok": True, "confirmed": False, "closed": False})
                return

            try:
                with runtime_lock:
                    left_dataset = runtime_box["left_dataset"]
                    right_dataset = runtime_box["right_dataset"]
                    image_state = runtime_box["image_state"]
                rows = _sanitize_rows(payload.get("rows"), image_state)
                result_payload = build_manual_measure_result_payload(left_dataset, right_dataset, image_state, rows)
                json_path, xlsx_path, latest_json_path, latest_xlsx_path = write_manual_measure_results(
                    result_payload,
                    output_dir=output_dir,
                    stem=output_stem or f"manual_tube_measure_{_utc_stamp()}",
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

            result_box["confirmed"] = True
            result_box["payload"] = result_payload
            result_box["json_path"] = json_path
            result_box["xlsx_path"] = xlsx_path
            result_box["latest_json_path"] = latest_json_path
            result_box["latest_xlsx_path"] = latest_xlsx_path
            done.set()
            self._send_json(
                {
                    "ok": True,
                    "confirmed": True,
                    "json_path": str(json_path),
                    "xlsx_path": str(xlsx_path),
                    "latest_json_path": str(latest_json_path),
                    "latest_xlsx_path": str(latest_xlsx_path),
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), ManualMeasureHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Medicion manual de tubos lista: {url}")
    print("La celda o script queda esperando hasta que presiones Confirmar o Cancelar.")
    _display_jupyter_link(url)
    if on_ready is not None:
        try:
            on_ready(url)
        except Exception as exc:
            print(f"Aviso: callback on_ready fallo: {exc}")
    if open_browser:
        try:
            webbrowser.open(url, new=1)
        except Exception:
            pass

    try:
        if not done.wait(timeout=timeout):
            print("Medicion manual de tubos: tiempo agotado sin resultado.")
            return ManualTubeMeasureRunResult(False, None, None, None, None, None)
        return ManualTubeMeasureRunResult(
            bool(result_box["confirmed"]),
            result_box["payload"],
            result_box["json_path"],
            result_box["xlsx_path"],
            result_box["latest_json_path"],
            result_box["latest_xlsx_path"],
        )
    finally:
        server.shutdown()
        server.server_close()
