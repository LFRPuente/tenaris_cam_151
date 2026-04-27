"""Local browser app for matching tube measurements from cam151 and cam152."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.sax.saxutils import escape as xml_escape
import zipfile

import cv2


BASE_CENTRAL_IN = 408.5
LEFT_REFERENCE_IN = 47.75
RIGHT_REFERENCE_IN = 41.0
MATCHER_INPUT_VERSION = 1
MATCHER_RESULT_VERSION = 1
_DIST_VIS_CELL_HINT = "show_bgr(dist_vis"
_WARP_ANALYSIS_CELL_HINT = "Warp recortado para analisis"
_WARP_HOMOGRAPHY_CELL_HINT = "Homografia: warp rectificado"


@dataclass
class TubeMatcherRunResult:
    confirmed: bool
    result_payload: dict[str, Any] | None
    result_json_path: Path | None
    result_xlsx_path: Path | None
    latest_json_path: Path | None
    latest_xlsx_path: Path | None


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _default_input_dir() -> Path:
    return _repo_root() / "artifacts" / "tube_matcher_inputs"


def _default_output_dir() -> Path:
    return _repo_root() / "artifacts" / "tube_matching"


def _candidate_input_dirs(input_dir: str | Path | None = None) -> list[Path]:
    if input_dir is not None:
        return [Path(input_dir)]

    repo_root = _repo_root()
    candidates = [
        _default_input_dir(),
        repo_root / "notebooks" / "artifacts" / "tube_matcher_inputs",
    ]
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _normalize_side(side: Any) -> str:
    raw = str(side or "").strip().lower()
    if raw in {"151", "cam151", "cam_151", "left", "izq", "izquierda"}:
        return "151"
    if raw in {"152", "cam152", "cam_152", "right", "der", "derecha"}:
        return "152"
    raise ValueError(f"Lado no soportado para matcher: {side!r}")


def _safe_name(text: Any, fallback: str) -> str:
    src = str(text or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in src)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return float(number)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _display_jupyter_link(url: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except Exception:
        return
    shell = get_ipython()
    if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
        return
    display(HTML(f'<a href="{url}" target="_blank">Abrir matcher de tubos</a>'))


def _clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _normalize_measurement_item(side: str, raw_item: Any, position: int) -> dict[str, Any]:
    if not isinstance(raw_item, dict):
        raise TypeError(f"Medicion invalida en posicion {position}: se esperaba dict.")

    tube_idx = _int_or_none(raw_item.get("tube_idx"))
    if tube_idx is None:
        tube_idx = position + 1

    distance_in = _float_or_none(raw_item.get("distance_in"))
    if distance_in is None:
        raise ValueError(f"Medicion {side}-{tube_idx} sin distance_in valido.")

    offset_px = _float_or_none(raw_item.get("offset_px"))
    raw_source_measurement = raw_item.get("source_measurement")
    if isinstance(raw_source_measurement, dict):
        source_measurement = _clone_json(raw_source_measurement)
    else:
        source_measurement = _clone_json(raw_item)
    if side == "152":
        raw_relative_position = str(raw_item.get("relative_position") or "").strip().lower()
        if raw_relative_position in {"before", "after"}:
            relative_position = raw_relative_position
        else:
            relative_position = "before" if offset_px is not None and offset_px < 0 else "after"
    else:
        relative_position = None

    item_id = f"{side}-{tube_idx:03d}-{position:03d}"
    return {
        "id": item_id,
        "side": side,
        "tube_idx": int(tube_idx),
        "display_name": f"Tubo {int(tube_idx)}",
        "distance_in": float(distance_in),
        "offset_px": offset_px,
        "offset_direction": (
            "negative"
            if offset_px is not None and offset_px < 0
            else "positive"
            if offset_px is not None and offset_px > 0
            else "zero"
            if offset_px is not None
            else None
        ),
        "relative_position": relative_position,
        "active": True,
        "unmatched": False,
        "notes": "",
        "source_order": int(position),
        "source_measurement": source_measurement,
    }


def build_matcher_input_dataset(
    side: str,
    tube_measurements: list[dict[str, Any]],
    *,
    image_path: str | None = None,
    roi_path: str | None = None,
    source_notebook: str | None = None,
    dataset_name: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    side_key = _normalize_side(side)
    items = [_normalize_measurement_item(side_key, raw_item, pos) for pos, raw_item in enumerate(tube_measurements or [])]
    return {
        "version": MATCHER_INPUT_VERSION,
        "side": side_key,
        "dataset_name": dataset_name or f"cam{side_key}",
        "generated_at": _iso_now(),
        "image_path": str(image_path) if image_path else None,
        "image_name": Path(image_path).name if image_path else None,
        "roi_path": str(roi_path) if roi_path else None,
        "source_notebook": source_notebook,
        "reference_in": LEFT_REFERENCE_IN if side_key == "151" else RIGHT_REFERENCE_IN,
        "base_central_in": BASE_CENTRAL_IN,
        "items": items,
        "extra_meta": _clone_json(extra_meta or {}),
    }


def export_tube_measurements(
    side: str,
    tube_measurements: list[dict[str, Any]],
    *,
    image_path: str | None = None,
    roi_path: str | None = None,
    source_notebook: str | None = None,
    dataset_name: str | None = None,
    extra_meta: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    dataset = build_matcher_input_dataset(
        side,
        tube_measurements,
        image_path=image_path,
        roi_path=roi_path,
        source_notebook=source_notebook,
        dataset_name=dataset_name,
        extra_meta=extra_meta,
    )
    side_key = dataset["side"]
    output_dir = Path(output_dir) if output_dir else _default_input_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = _utc_stamp()
    base_name = f"cam{side_key}_tube_measurements_{stamp}"
    path = output_dir / f"{base_name}.json"
    latest_path = output_dir / f"cam{side_key}_tube_measurements_latest.json"
    payload_text = json.dumps(dataset, indent=2, ensure_ascii=False)
    path.write_text(payload_text, encoding="utf-8")
    latest_path.write_text(payload_text, encoding="utf-8")
    return path


def find_latest_measurement_export(side: str, *, input_dir: str | Path | None = None) -> Path:
    side_key = _normalize_side(side)
    latest_candidates: list[Path] = []
    historical_candidates: list[Path] = []
    for candidate_dir in _candidate_input_dirs(input_dir):
        latest_path = candidate_dir / f"cam{side_key}_tube_measurements_latest.json"
        if latest_path.exists():
            latest_candidates.append(latest_path)
        historical_candidates.extend(candidate_dir.glob(f"cam{side_key}_tube_measurements_*.json"))
    if latest_candidates:
        return max(latest_candidates, key=lambda p: p.stat().st_mtime)
    candidates = sorted(historical_candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        dirs_txt = ", ".join(str(path) for path in _candidate_input_dirs(input_dir))
        raise FileNotFoundError(f"No se encontro export de mediciones para cam{side_key} en {dirs_txt}")
    return candidates[0]


def _load_dataset(source: str | Path | dict[str, Any], expected_side: str | None = None) -> dict[str, Any]:
    if isinstance(source, dict):
        payload = deepcopy(source)
    else:
        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_input_path"] = str(path)

    if not isinstance(payload, dict):
        raise TypeError("Dataset de matcher invalido.")
    side = _normalize_side(payload.get("side"))
    if expected_side is not None and side != _normalize_side(expected_side):
        raise ValueError(f"Se esperaba dataset {expected_side}, se recibio {side}.")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Dataset de matcher sin lista items.")

    normalized_items = [_normalize_measurement_item(side, item, pos) for pos, item in enumerate(items)]
    for pos, item in enumerate(normalized_items):
        src_item = items[pos] if isinstance(items[pos], dict) else {}
        item["active"] = bool(src_item.get("active", True))
        item["unmatched"] = bool(src_item.get("unmatched", False))
        notes = str(src_item.get("notes") or "").strip()
        item["notes"] = notes[:200]
        if side == "152":
            rel = str(src_item.get("relative_position") or item["relative_position"] or "").strip().lower()
            item["relative_position"] = rel if rel in {"before", "after"} else item["relative_position"]

    payload["side"] = side
    payload["items"] = normalized_items
    payload["reference_in"] = float(payload.get("reference_in") or (LEFT_REFERENCE_IN if side == "151" else RIGHT_REFERENCE_IN))
    payload["base_central_in"] = float(payload.get("base_central_in") or BASE_CENTRAL_IN)
    return payload


def _sanitize_client_items(client_items: Any, source_items: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    if not isinstance(client_items, list):
        raise ValueError(f"Payload {side} invalido: items debe ser lista.")
    source_map = {item["id"]: item for item in source_items}
    received_ids = []
    sanitized: list[dict[str, Any]] = []
    for raw in client_items:
        if not isinstance(raw, dict):
            raise ValueError(f"Payload {side} invalido: item no es dict.")
        item_id = str(raw.get("id") or "")
        if item_id not in source_map:
            raise ValueError(f"Payload {side} invalido: id desconocido {item_id!r}.")
        if item_id in received_ids:
            raise ValueError(f"Payload {side} invalido: id repetido {item_id!r}.")
        received_ids.append(item_id)
        item = deepcopy(source_map[item_id])
        item["active"] = bool(raw.get("active", item["active"]))
        item["unmatched"] = bool(raw.get("unmatched", item["unmatched"]))
        item["notes"] = str(raw.get("notes") or item.get("notes") or "").strip()[:200]
        if side == "152":
            rel = str(raw.get("relative_position") or item.get("relative_position") or "").strip().lower()
            item["relative_position"] = rel if rel in {"before", "after"} else item.get("relative_position") or "after"
        sanitized.append(item)
    if set(received_ids) != set(source_map):
        raise ValueError(f"Payload {side} invalido: faltan items en la actualizacion.")
    return sanitized


def _compute_r151(item: dict[str, Any] | None) -> float | None:
    if not item:
        return None
    dist = _float_or_none(item.get("distance_in"))
    if dist is None:
        return None
    return float(dist - LEFT_REFERENCE_IN)


def _compute_r152(item: dict[str, Any] | None) -> float | None:
    if not item:
        return None
    dist = _float_or_none(item.get("distance_in"))
    if dist is None:
        return None
    rel = str(item.get("relative_position") or "").strip().lower()
    if rel == "before":
        return float(RIGHT_REFERENCE_IN - dist)
    return float(RIGHT_REFERENCE_IN + dist)


def _build_result_row(row_number: int, left_item: dict[str, Any] | None, right_item: dict[str, Any] | None, status: str) -> dict[str, Any]:
    r151 = _compute_r151(left_item)
    r152 = _compute_r152(right_item)
    total = None
    if r151 is not None and r152 is not None:
        total = float(BASE_CENTRAL_IN + r151 + r152)

    observations = []
    if left_item and left_item.get("notes"):
        observations.append(f"151: {left_item['notes']}")
    if right_item and right_item.get("notes"):
        observations.append(f"152: {right_item['notes']}")
    if status == "left_only_manual":
        observations.append("Sin match manual en 151")
    elif status == "right_only_manual":
        observations.append("Sin match manual en 152")
    elif status == "left_only":
        observations.append("Falta medicion del lado 152")
    elif status == "right_only":
        observations.append("Falta medicion del lado 151")

    return {
        "tube_number": int(row_number),
        "match_status": status,
        "tube_idx_151": left_item.get("tube_idx") if left_item else None,
        "active_151": bool(left_item.get("active")) if left_item else False,
        "measurement_151": left_item.get("distance_in") if left_item else None,
        "r151": r151,
        "tube_idx_152": right_item.get("tube_idx") if right_item else None,
        "active_152": bool(right_item.get("active")) if right_item else False,
        "measurement_152": right_item.get("distance_in") if right_item else None,
        "r152_position": right_item.get("relative_position") if right_item else None,
        "r152": r152,
        "total_length_in": total,
        "observations": " | ".join(observations) if observations else "",
    }


def build_match_rows(left_items: list[dict[str, Any]], right_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    left_queue = [deepcopy(item) for item in left_items if item.get("active")]
    right_queue = [deepcopy(item) for item in right_items if item.get("active")]
    rows: list[dict[str, Any]] = []
    left_pos = 0
    right_pos = 0
    row_number = 1

    while left_pos < len(left_queue) or right_pos < len(right_queue):
        left_item = left_queue[left_pos] if left_pos < len(left_queue) else None
        right_item = right_queue[right_pos] if right_pos < len(right_queue) else None

        if left_item is not None and left_item.get("unmatched"):
            rows.append(_build_result_row(row_number, left_item, None, "left_only_manual"))
            row_number += 1
            left_pos += 1
            continue

        if right_item is not None and right_item.get("unmatched"):
            rows.append(_build_result_row(row_number, None, right_item, "right_only_manual"))
            row_number += 1
            right_pos += 1
            continue

        if left_item is not None and right_item is not None:
            rows.append(_build_result_row(row_number, left_item, right_item, "matched"))
            row_number += 1
            left_pos += 1
            right_pos += 1
            continue

        if left_item is not None:
            rows.append(_build_result_row(row_number, left_item, None, "left_only"))
            row_number += 1
            left_pos += 1
            continue

        rows.append(_build_result_row(row_number, None, right_item, "right_only"))
        row_number += 1
        right_pos += 1

    return rows


def build_result_payload(
    left_dataset: dict[str, Any],
    right_dataset: dict[str, Any],
    left_items: list[dict[str, Any]],
    right_items: list[dict[str, Any]],
) -> dict[str, Any]:
    match_rows = build_match_rows(left_items, right_items)
    summary = {
        "matched": sum(1 for row in match_rows if row["match_status"] == "matched"),
        "left_only": sum(1 for row in match_rows if row["match_status"] in {"left_only", "left_only_manual"}),
        "right_only": sum(1 for row in match_rows if row["match_status"] in {"right_only", "right_only_manual"}),
        "inactive_151": sum(1 for item in left_items if not item.get("active")),
        "inactive_152": sum(1 for item in right_items if not item.get("active")),
        "manual_unmatched_151": sum(1 for item in left_items if item.get("active") and item.get("unmatched")),
        "manual_unmatched_152": sum(1 for item in right_items if item.get("active") and item.get("unmatched")),
    }
    return {
        "version": MATCHER_RESULT_VERSION,
        "generated_at": _iso_now(),
        "config": {
            "base_central_in": BASE_CENTRAL_IN,
            "left_reference_in": LEFT_REFERENCE_IN,
            "right_reference_in": RIGHT_REFERENCE_IN,
        },
        "inputs": {
            "cam151": {
                "dataset_name": left_dataset.get("dataset_name"),
                "image_name": left_dataset.get("image_name"),
                "image_path": left_dataset.get("image_path"),
                "roi_path": left_dataset.get("roi_path"),
                "source_notebook": left_dataset.get("source_notebook"),
                "input_path": left_dataset.get("_input_path"),
            },
            "cam152": {
                "dataset_name": right_dataset.get("dataset_name"),
                "image_name": right_dataset.get("image_name"),
                "image_path": right_dataset.get("image_path"),
                "roi_path": right_dataset.get("roi_path"),
                "source_notebook": right_dataset.get("source_notebook"),
                "input_path": right_dataset.get("_input_path"),
            },
        },
        "summary": summary,
        "left_items": left_items,
        "right_items": right_items,
        "rows": match_rows,
    }


def _copy_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_source_notebook_path(source_notebook: Any) -> Path | None:
    raw = str(source_notebook or "").strip()
    if not raw:
        return None
    raw_path = Path(raw)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        repo_root = _repo_root()
        candidates.extend(
            [
                Path.cwd() / raw_path,
                Path.cwd() / "notebooks" / raw_path,
                repo_root / raw_path,
                repo_root / "notebooks" / raw_path,
            ]
        )
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _decode_notebook_png(raw_image: Any) -> bytes | None:
    if isinstance(raw_image, list):
        raw_image = "".join(str(part) for part in raw_image)
    if not raw_image:
        return None
    try:
        return base64.b64decode(raw_image)
    except Exception:
        return None


def _extract_notebook_png_by_hints(source_notebook: Any, preferred_hints: list[str]) -> bytes | None:
    notebook_path = _resolve_source_notebook_path(source_notebook)
    if notebook_path is None:
        return None
    try:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    fallback_png: bytes | None = None
    preferred_hints = [str(hint or "") for hint in preferred_hints if str(hint or "")]
    cells = [cell for cell in payload.get("cells", []) if isinstance(cell, dict)]

    for hint in preferred_hints:
        for cell in cells:
            cell_source = "".join(cell.get("source", []))
            if hint not in cell_source:
                continue
            for output in cell.get("outputs", []):
                if not isinstance(output, dict):
                    continue
                data = output.get("data")
                if not isinstance(data, dict):
                    continue
                png_bytes = _decode_notebook_png(data.get("image/png"))
                if png_bytes:
                    return png_bytes

    for cell in cells:
        for output in cell.get("outputs", []):
            if not isinstance(output, dict):
                continue
            data = output.get("data")
            if not isinstance(data, dict):
                continue
            png_bytes = _decode_notebook_png(data.get("image/png"))
            if not png_bytes:
                continue
            fallback_png = png_bytes
    return fallback_png


def _png_dimensions(png_bytes: bytes | None) -> tuple[int, int] | tuple[None, None]:
    if not png_bytes or len(png_bytes) < 24:
        return None, None
    if png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    width = int.from_bytes(png_bytes[16:20], "big")
    height = int.from_bytes(png_bytes[20:24], "big")
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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


def _load_artifact_warp_image(side: str, dataset: dict[str, Any]) -> tuple[bytes, str, int, int] | None:
    image_name = str(dataset.get("image_name") or "").strip()
    if not image_name:
        return None
    image_stem = Path(image_name).stem
    repo_root = _repo_root()
    candidates = [
        repo_root / "artifacts" / f"notebook_step_by_step_cam{side}" / image_stem / "01_homography" / "homography_warp.jpg",
        repo_root / "artifacts" / f"notebook_step_by_step_cam{side}" / image_stem / "01_homography_base_for_full_view" / "homography_warp.jpg",
        repo_root / "artifacts" / f"notebook_step_by_step_cam{side}" / image_stem / "01_homography_full" / "homography_warp.jpg",
    ]
    for path in candidates:
        if not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            continue
        height, width = image.shape[:2]
        return path.read_bytes(), "image/jpeg", int(width), int(height)
    return None


def _detection_roi_origin(dataset: dict[str, Any], width: int, height: int) -> tuple[float, float]:
    extra_meta = dataset.get("extra_meta") or {}
    if not isinstance(extra_meta, dict):
        return 0.0, 0.0
    raw_roi = extra_meta.get("detection_roi")
    if not isinstance(raw_roi, list) or len(raw_roi) < 4:
        return 0.0, 0.0
    x0 = _float_or_none(raw_roi[0])
    y0 = _float_or_none(raw_roi[1])
    roi_w = _float_or_none(raw_roi[2])
    roi_h = _float_or_none(raw_roi[3])
    if x0 is None or y0 is None or roi_w is None or roi_h is None:
        return 0.0, 0.0
    if abs(float(roi_w) - float(width)) <= 2.0 and abs(float(roi_h) - float(height)) <= 2.0:
        return float(x0), float(y0)
    return 0.0, 0.0


def _build_viewer_markers(dataset: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    items = dataset.get("items") or []
    ordered_items = sorted(
        [item for item in items if isinstance(item, dict)],
        key=lambda item: (int(item.get("source_order", 0)), int(item.get("tube_idx", 0))),
    )
    if not ordered_items:
        return []
    crop_x0, crop_y0 = _detection_roi_origin(dataset, width, height)

    top_pad = max(42.0, height * 0.07)
    bottom_pad = max(42.0, height * 0.07)
    if len(ordered_items) == 1:
        estimated_ys = [0.5 * height]
    else:
        step = (height - top_pad - bottom_pad) / max(len(ordered_items) - 1, 1)
        estimated_ys = [height - bottom_pad - (idx * step) for idx in range(len(ordered_items))]

    markers: list[dict[str, Any]] = []
    for index, item in enumerate(ordered_items):
        source_measurement = item.get("source_measurement")
        if not isinstance(source_measurement, dict):
            source_measurement = {}

        x_value = None
        for key in ("x_start_raw_warp", "x_start_warp", "x_start_smooth_warp", "ref_x"):
            x_value = _float_or_none(source_measurement.get(key))
            if x_value is not None:
                break
        if x_value is None:
            x_value = 0.5 * width
        else:
            x_value = float(x_value) - crop_x0

        y_value = _float_or_none(source_measurement.get("y_center_warp"))
        if y_value is None:
            y_value = estimated_ys[index]
        else:
            y_value = float(y_value) - crop_y0

        ref_x = _float_or_none(source_measurement.get("ref_x"))
        if ref_x is not None:
            ref_x = float(ref_x) - crop_x0

        markers.append(
            {
                "id": item.get("id"),
                "side": item.get("side"),
                "tube_idx": item.get("tube_idx"),
                "distance_in": item.get("distance_in"),
                "x": round(_clamp(float(x_value), 18.0, max(18.0, width - 18.0)), 3),
                "y": round(_clamp(float(y_value), 18.0, max(18.0, height - 18.0)), 3),
                "ref_x": ref_x,
                "offset_px": _float_or_none(source_measurement.get("offset_px")),
            }
        )
    return markers


def _build_image_viewer_state(side: str, dataset: dict[str, Any], asset_url: str) -> dict[str, Any]:
    asset_bytes: bytes | None = None
    asset_content_type = "image/png"
    width: int | None = None
    height: int | None = None

    exported_image_path = _resolve_dataset_image_path(dataset)
    if exported_image_path is not None:
        image = cv2.imread(str(exported_image_path), cv2.IMREAD_COLOR)
        if image is not None and image.size > 0:
            height, width = image.shape[:2]
            asset_bytes = exported_image_path.read_bytes()
            suffix = exported_image_path.suffix.lower()
            asset_content_type = "image/png" if suffix == ".png" else "image/jpeg"

    if asset_bytes is None:
        artifact = _load_artifact_warp_image(side, dataset)
        if artifact is not None:
            asset_bytes, asset_content_type, width, height = artifact
        else:
            png_bytes = _extract_notebook_png_by_hints(
                dataset.get("source_notebook"),
                [
                    _WARP_ANALYSIS_CELL_HINT,
                    _WARP_HOMOGRAPHY_CELL_HINT,
                ],
            )
            width, height = _png_dimensions(png_bytes)
            asset_bytes = png_bytes
            asset_content_type = "image/png"

    if not asset_bytes or width is None or height is None:
        return {
            "available": False,
            "url": None,
            "width": None,
            "height": None,
            "markers": [],
            "source_notebook": dataset.get("source_notebook"),
            "image_name": dataset.get("image_name"),
            "asset_bytes": None,
            "asset_content_type": None,
        }

    return {
        "available": True,
        "url": asset_url,
        "width": width,
        "height": height,
        "markers": _build_viewer_markers(dataset, width, height),
        "source_notebook": dataset.get("source_notebook"),
        "image_name": dataset.get("image_name"),
        "asset_bytes": asset_bytes,
        "asset_content_type": asset_content_type,
    }


def _xlsx_column_name(index: int) -> str:
    value = int(index)
    out = ""
    while value >= 0:
        value, rem = divmod(value, 26)
        out = chr(65 + rem) + out
        value -= 1
    return out


def _excel_serial_date(iso_text: str) -> float | None:
    try:
        dt = datetime.fromisoformat(iso_text.replace("Z", "+00:00"))
    except ValueError:
        return None
    epoch = datetime(1899, 12, 30, tzinfo=dt.tzinfo)
    delta = dt - epoch
    return float(delta.days) + float(delta.seconds) / 86400.0


def _sheet_xml(sheet_name: str, headers: list[str], rows: list[dict[str, Any]]) -> str:
    col_count = max(len(headers), 1)
    row_xml: list[str] = []

    def cell_ref(col_index: int, row_index: int) -> str:
        return f"{_xlsx_column_name(col_index)}{row_index}"

    def inline_cell(ref: str, text: str) -> str:
        safe = xml_escape(text)
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{safe}</t></is></c>'

    def number_cell(ref: str, number: float) -> str:
        return f'<c r="{ref}"><v>{number}</v></c>'

    header_cells = [inline_cell(cell_ref(idx, 1), header) for idx, header in enumerate(headers)]
    row_xml.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_index, row in enumerate(rows, start=2):
        cells: list[str] = []
        for col_index, header in enumerate(headers):
            ref = cell_ref(col_index, row_index)
            value = row.get(header)
            if value is None:
                continue
            if isinstance(value, bool):
                cells.append(number_cell(ref, 1 if value else 0))
            elif isinstance(value, (int, float)):
                cells.append(number_cell(ref, float(value)))
            else:
                text = str(value)
                serial = _excel_serial_date(text) if header.endswith("_at") else None
                if serial is not None:
                    cells.append(number_cell(ref, serial))
                else:
                    cells.append(inline_cell(ref, text))
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    dimension = f"A1:{_xlsx_column_name(col_count - 1)}{max(len(rows) + 1, 1)}"
    cols_xml = "".join(f'<col min="{idx+1}" max="{idx+1}" width="18" customWidth="1"/>' for idx in range(col_count))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '<selection pane="bottomLeft" activeCell="A2" sqref="A2"/></sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f"<cols>{cols_xml}</cols>"
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '<autoFilter ref="A1:' + _xlsx_column_name(col_count - 1) + str(max(len(rows) + 1, 1)) + '"/>'
        "</worksheet>"
    )


def _write_simple_xlsx(path: Path, sheets: list[tuple[str, list[dict[str, Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not sheets:
        raise ValueError("Se requiere al menos una hoja para exportar xlsx.")

    headers_by_sheet: list[list[str]] = []
    for _name, rows in sheets:
        header_order: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in header_order:
                    header_order.append(key)
        if not header_order:
            header_order = ["empty"]
        headers_by_sheet.append(header_order)

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for idx in range(len(sheets)):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx + 1}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx, (sheet_name, _rows) in enumerate(sheets, start=1):
        workbook_sheets.append(
            f'<sheet name="{xml_escape(sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    workbook_rels.append("</Relationships>")
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<bookViews><workbookView activeTab=\"0\"/></bookViews>"
        f"<sheets>{''.join(workbook_sheets)}</sheets>"
        "</workbook>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
        "<fills count=\"2\"><fill><patternFill patternType=\"none\"/></fill><fill><patternFill patternType=\"gray125\"/></fill></fills>"
        "<borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
        "<cellXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyNumberFormat=\"0\"/></cellXfs>"
        "<cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>"
        "</styleSheet>"
    )

    now_iso = _iso_now()
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:creator>OpenAI Codex</dc:creator>"
        f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{xml_escape(now_iso)}</dcterms:created>"
        f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{xml_escape(now_iso)}</dcterms:modified>"
        "</cp:coreProperties>"
    )
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>OpenAI Codex</Application>"
        f"<HeadingPairs><vt:vector size=\"2\" baseType=\"variant\"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>{len(sheets)}</vt:i4></vt:variant></vt:vector></HeadingPairs>"
        f"<TitlesOfParts><vt:vector size=\"{len(sheets)}\" baseType=\"lpstr\">"
        + "".join(f"<vt:lpstr>{xml_escape(name)}</vt:lpstr>" for name, _rows in sheets)
        + "</vt:vector></TitlesOfParts>"
        "</Properties>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        zf.writestr("xl/styles.xml", styles_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)
        for idx, (sheet_name, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _sheet_xml(sheet_name, headers_by_sheet[idx - 1], rows))


def write_match_results(
    result_payload: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    stem: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(output_dir) if output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    name_stem = _safe_name(stem, f"tube_match_result_{stamp}")
    json_path = output_dir / f"{name_stem}.json"
    xlsx_path = output_dir / f"{name_stem}.xlsx"
    latest_json_path = output_dir / "tube_match_latest.json"
    latest_xlsx_path = output_dir / "tube_match_latest.xlsx"

    json_text = json.dumps(result_payload, indent=2, ensure_ascii=False)
    _copy_text(json_path, json_text)
    _copy_text(latest_json_path, json_text)

    match_rows = list(result_payload.get("rows") or [])
    audit_rows: list[dict[str, Any]] = []
    for side_key, items_key in (("151", "left_items"), ("152", "right_items")):
        for item in result_payload.get(items_key, []):
            audit_rows.append(
                {
                    "side": side_key,
                    "tube_idx": item.get("tube_idx"),
                    "active": item.get("active"),
                    "unmatched": item.get("unmatched"),
                    "distance_in": item.get("distance_in"),
                    "relative_position": item.get("relative_position"),
                    "notes": item.get("notes"),
                    "source_order": item.get("source_order"),
                }
            )
    _write_simple_xlsx(xlsx_path, [("matches", match_rows), ("audit", audit_rows)])
    _write_simple_xlsx(latest_xlsx_path, [("matches", match_rows), ("audit", audit_rows)])
    return json_path, xlsx_path, latest_json_path, latest_xlsx_path


def _initial_state_for_html(
    left_dataset: dict[str, Any],
    right_dataset: dict[str, Any],
    image_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config": {
            "base_central_in": BASE_CENTRAL_IN,
            "left_reference_in": LEFT_REFERENCE_IN,
            "right_reference_in": RIGHT_REFERENCE_IN,
        },
        "left": {
            "dataset_name": left_dataset.get("dataset_name"),
            "image_name": left_dataset.get("image_name"),
            "items": left_dataset.get("items", []),
        },
        "right": {
            "dataset_name": right_dataset.get("dataset_name"),
            "image_name": right_dataset.get("image_name"),
            "items": right_dataset.get("items", []),
        },
        "images": image_state,
    }


def _matcher_html(left_dataset: dict[str, Any], right_dataset: dict[str, Any], image_state: dict[str, Any]) -> str:
    template_path = Path(__file__).with_name("tube_matcher.html")
    html_template = template_path.read_text(encoding="utf-8")
    initial_state = _initial_state_for_html(left_dataset, right_dataset, image_state)
    return html_template.replace("__INITIAL_STATE__", json.dumps(initial_state, ensure_ascii=False))


def run_tube_matcher(
    cam151_input: str | Path | dict[str, Any] | None = None,
    cam152_input: str | Path | dict[str, Any] | None = None,
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_stem: str | None = None,
    timeout: float | None = None,
    open_browser: bool = True,
    on_ready: Any | None = None,
) -> TubeMatcherRunResult:
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
            "151": _build_image_viewer_state("151", left_dataset, "/asset/cam151-final.jpg"),
            "152": _build_image_viewer_state("152", right_dataset, "/asset/cam152-final.jpg"),
        }
        asset_map: dict[str, tuple[bytes, str]] = {}
        for image_meta in image_state.values():
            asset_bytes = image_meta.pop("asset_bytes", None)
            asset_content_type = image_meta.pop("asset_content_type", None)
            asset_url = image_meta.get("url")
            if asset_bytes and asset_url:
                asset_map[str(asset_url)] = (asset_bytes, str(asset_content_type or "application/octet-stream"))
        html = _matcher_html(left_dataset, right_dataset, image_state)
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

    class MatcherHandler(BaseHTTPRequestHandler):
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
                left_items = _sanitize_client_items(payload.get("leftItems"), left_dataset["items"], "151")
                right_items = _sanitize_client_items(payload.get("rightItems"), right_dataset["items"], "152")
                result_payload = build_result_payload(left_dataset, right_dataset, left_items, right_items)
                json_path, xlsx_path, latest_json_path, latest_xlsx_path = write_match_results(
                    result_payload,
                    output_dir=output_dir,
                    stem=output_stem or f"tube_match_result_{_utc_stamp()}",
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

    server = ThreadingHTTPServer(("127.0.0.1", 0), MatcherHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Matcher de tubos listo: {url}")
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
            print("Matcher de tubos: tiempo agotado sin resultado.")
            return TubeMatcherRunResult(False, None, None, None, None, None)
        return TubeMatcherRunResult(
            bool(result_box["confirmed"]),
            result_box["payload"],
            result_box["json_path"],
            result_box["xlsx_path"],
            result_box["latest_json_path"],
            result_box["latest_xlsx_path"],
        )
    except KeyboardInterrupt:
        print("Matcher de tubos cancelado desde el kernel.")
        return TubeMatcherRunResult(False, None, None, None, None, None)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
