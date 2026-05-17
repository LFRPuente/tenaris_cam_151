"""Client-facing MVP viewer for sorting table plus raw camera overlays."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .capture_history import (
    CaptureRunError,
    capture_and_process_pair,
    delete_capture_run,
    list_capture_run_manifests,
    load_capture_run_manifest,
    load_latest_capture_run_manifest,
)
from .homography_preview import build_homography_preview
from .homography_profiles import (
    list_profiles as list_homography_profiles,
    resolve_profile_path as resolve_homography_profile_path,
    save_homography_profile,
    set_default_profile as set_default_homography_profile,
)
from .roi_store import load_rois
from .tube_matcher_proc import (
    _load_dataset,
    _normalize_side,
    _repo_root,
    find_latest_measurement_export,
)


MATCH_RESULT_VERSION = 1
_MATCH_STAMP_RE = re.compile(r"cam(?P<side>151|152)_tube_measurements_(?P<stamp>\d{8}_\d{6})\.json$", re.IGNORECASE)
_ASSET_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass
class SortingTableMvpServerHandle:
    url: str
    server: ThreadingHTTPServer
    thread: threading.Thread

    def wait(self) -> None:
        try:
            self.thread.join()
        except KeyboardInterrupt:
            self.close()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _display_jupyter_link(url: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except Exception:
        return
    shell = get_ipython()
    if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
        return
    display(HTML(f'<a href="{url}" target="_blank">Open Sorting Table MVP</a>'))


def _default_match_dir() -> Path:
    return _repo_root() / "artifacts" / "tube_matching"


def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_iso_datetime(raw_value: Any) -> datetime | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _stamp_datetime_from_path(path: Path) -> datetime | None:
    match = _MATCH_STAMP_RE.search(path.name)
    if not match:
        return None
    stamp = match.group("stamp")
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _image_mime_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg", ".jfif"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _asset_token(value: Any, *, fallback: str) -> str:
    cleaned = _ASSET_TOKEN_RE.sub("_", str(value or "").strip()).strip("_")
    return cleaned or fallback


def _resolve_manifest_match_source(manifest: dict[str, Any]) -> Path:
    processing = dict(manifest.get("processing") or {})
    for candidate in (
        processing.get("match_latest_json_path"),
        processing.get("match_json_path"),
    ):
        path = _resolve_existing_path(candidate)
        if path is not None:
            return path
    raise FileNotFoundError(f"Run {manifest.get('run_id')!r} does not have a usable matching result.")


def _run_state_from_manifest(manifest: dict[str, Any] | None, *, latest_run_id: str | None = None) -> dict[str, Any] | None:
    if not isinstance(manifest, dict):
        return None
    processing = dict(manifest.get("processing") or {})
    summary = dict(processing.get("summary") or {})
    pipe_end_yolo = dict(processing.get("pipe_end_yolo") or {})
    tube_counts = dict(processing.get("tube_counts") or {})
    run_id = str(manifest.get("run_id") or "").strip()
    return {
        "run_id": run_id,
        "captured_at": str(manifest.get("captured_at") or ""),
        "status": str(manifest.get("status") or ""),
        "summary": {
            "matched": int(summary.get("matched") or 0),
            "left_only": int(summary.get("left_only") or 0),
            "right_only": int(summary.get("right_only") or 0),
        },
        "detection_source": str(processing.get("detection_source") or ""),
        "tube_counts": {
            "cam151": int(tube_counts.get("cam151") or 0),
            "cam152": int(tube_counts.get("cam152") or 0),
        },
        "pipe_end_yolo": pipe_end_yolo,
        "cam151_image_name": str((((manifest.get("cameras") or {}).get("cam151") or {}).get("image_name")) or ""),
        "cam152_image_name": str((((manifest.get("cameras") or {}).get("cam152") or {}).get("image_name")) or ""),
        "error": str(manifest.get("error") or ""),
        "is_latest": bool(run_id and latest_run_id and run_id == latest_run_id),
    }


def _history_entry_from_manifest(manifest: dict[str, Any], *, latest_run_id: str | None = None) -> dict[str, Any]:
    run_state = _run_state_from_manifest(manifest, latest_run_id=latest_run_id) or {}
    processing = dict(manifest.get("processing") or {})
    can_open = _resolve_existing_path(processing.get("match_latest_json_path") or processing.get("match_json_path")) is not None
    captured_at = str(manifest.get("captured_at") or "")
    return {
        **run_state,
        "can_open": bool(can_open),
        "captured_date": captured_at[:10] if captured_at else "",
        "run_url": f"/?run_id={run_state.get('run_id')}" if can_open and run_state.get("run_id") else None,
    }


def _legacy_match_artifact_dir() -> Path:
    return _default_match_dir()


def _resolve_legacy_match_artifact(artifact_name: str) -> Path:
    candidate = _legacy_match_artifact_dir() / Path(str(artifact_name or "")).name
    if not candidate.exists():
        raise FileNotFoundError(f"No existe artefacto historico: {artifact_name}")
    return candidate


def _history_entry_from_legacy_match(path: Path) -> dict[str, Any] | None:
    if path.name.lower() == "tube_match_latest.json":
        return None
    payload = _load_match_result(path)
    summary = dict(payload.get("summary") or {})
    inputs = dict(payload.get("inputs") or {})
    generated_at = str(payload.get("generated_at") or "")
    artifact_name = path.name
    artifact_id = f"artifact:{path.stem}"
    return {
        "run_id": artifact_id,
        "captured_at": generated_at,
        "status": "imported_artifact",
        "summary": {
            "matched": int(summary.get("matched") or 0),
            "left_only": int(summary.get("left_only") or 0),
            "right_only": int(summary.get("right_only") or 0),
        },
        "cam151_image_name": str(((inputs.get("cam151") or {}).get("image_name")) or ""),
        "cam152_image_name": str(((inputs.get("cam152") or {}).get("image_name")) or ""),
        "error": "",
        "is_latest": False,
        "can_open": True,
        "captured_date": generated_at[:10] if generated_at else "",
        "run_url": f"/?artifact={artifact_name}",
        "artifact_name": artifact_name,
    }


def _list_legacy_history_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(_legacy_match_artifact_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            entry = _history_entry_from_legacy_match(path)
        except Exception:
            continue
        if entry is not None:
            entries.append(entry)
    return entries


def _is_cam152_image(image_path: str | Path) -> bool:
    # Alineado con el notebook/backend: la variante base cam_152.jpeg ya viene espejada,
    # pero los pares cam_152_* necesitan mirror adicional dentro del pipeline.
    stem = Path(image_path).stem.lower()
    return stem.startswith("cam_152_") or stem.startswith("cam152_")


def _default_manual_roi_path(image_name: str | None) -> Path | None:
    if not image_name:
        return None
    stem = Path(image_name).stem
    candidate = _repo_root() / "manual_rois" / f"{stem}_rois.toml"
    return candidate if candidate.exists() else None


def _resolve_existing_path(raw_value: Any) -> Path | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    repo_root = _repo_root()
    candidates = [
        repo_root / text,
        repo_root / "test_images" / Path(text).name,
        repo_root / "manual_rois" / Path(text).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_latest_match_result(*, output_dir: str | Path | None = None) -> Path:
    output_root = Path(output_dir) if output_dir else _default_match_dir()
    latest_path = output_root / "tube_match_latest.json"
    if latest_path.exists():
        return latest_path
    candidates = sorted(output_root.glob("tube_match_result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Could not find tube_match_latest.json or any historical results under {output_root}")
    return candidates[0]


def _load_match_result(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        payload = json.loads(json.dumps(source))
    else:
        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_input_path"] = str(path)
    if not isinstance(payload, dict):
        raise TypeError("Resultado de matching invalido.")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("El resultado de matching no contiene rows.")
    payload["version"] = int(payload.get("version") or MATCH_RESULT_VERSION)
    return payload


def _resolve_dataset_from_match(
    match_payload: dict[str, Any],
    side: str,
    source_override: str | Path | dict[str, Any] | None = None,
) -> dict[str, Any]:
    side_key = _normalize_side(side)
    if source_override is not None:
        return _load_dataset(source_override, expected_side=side_key)

    side_info = ((match_payload.get("inputs") or {}).get(f"cam{side_key}") or {})
    input_path = _resolve_existing_path(side_info.get("input_path"))
    if input_path is not None and input_path.name.lower() != f"cam{side_key}_tube_measurements_latest.json":
        return _load_dataset(input_path, expected_side=side_key)

    match_dt = _parse_iso_datetime(match_payload.get("generated_at"))
    search_dir = input_path.parent if input_path is not None else (_repo_root() / "artifacts" / "tube_matcher_inputs")
    historical = []
    for candidate in search_dir.glob(f"cam{side_key}_tube_measurements_*.json"):
        stamp_dt = _stamp_datetime_from_path(candidate)
        if stamp_dt is None:
            continue
        historical.append((stamp_dt, candidate))
    if historical and match_dt is not None:
        eligible = [candidate for stamp_dt, candidate in historical if stamp_dt <= match_dt]
        if eligible:
            chosen = max(eligible, key=lambda path: path.stat().st_mtime)
            return _load_dataset(chosen, expected_side=side_key)
    if historical:
        chosen = max((candidate for _stamp_dt, candidate in historical), key=lambda path: path.stat().st_mtime)
        return _load_dataset(chosen, expected_side=side_key)
    if input_path is not None and input_path.exists():
        return _load_dataset(input_path, expected_side=side_key)
    fallback = find_latest_measurement_export(side_key, input_dir=search_dir)
    return _load_dataset(fallback, expected_side=side_key)


def _resolve_raw_image_path(dataset: dict[str, Any], match_payload: dict[str, Any], side: str) -> Path:
    side_key = _normalize_side(side)
    side_info = ((match_payload.get("inputs") or {}).get(f"cam{side_key}") or {})
    for candidate in (
        dataset.get("image_path"),
        side_info.get("image_path"),
    ):
        path = _resolve_existing_path(candidate)
        if path is not None:
            return path
    image_name = str(dataset.get("image_name") or side_info.get("image_name") or "").strip()
    fallback = _repo_root() / "test_images" / image_name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find a RAW image for cam{side_key}: {image_name or 'unnamed'}")


def _resolve_roi_path(dataset: dict[str, Any], match_payload: dict[str, Any], side: str) -> Path:
    side_key = _normalize_side(side)
    side_info = ((match_payload.get("inputs") or {}).get(f"cam{side_key}") or {})
    for candidate in (
        dataset.get("roi_path"),
        side_info.get("roi_path"),
    ):
        path = _resolve_existing_path(candidate)
        if path is not None:
            return path
    fallback = _default_manual_roi_path(str(dataset.get("image_name") or side_info.get("image_name") or ""))
    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"Could not find a manual ROI for cam{side_key}.")


def _build_inverse_output_transform(image_path: Path, roi_payload: dict[str, Any], output_dir: Path) -> tuple[np.ndarray, tuple[int, int], bool]:
    mirror_in_backend = _is_cam152_image(image_path)
    preview = build_homography_preview(
        image_path=image_path,
        lines=list(roi_payload.get("lines") or []),
        points=list(roi_payload.get("points") or []),
        output_dir=output_dir,
        src_points_override=roi_payload.get("src_points_override"),
        dst_rect_override=roi_payload.get("dst_rect_override"),
        flip_horizontal=mirror_in_backend,
        output_flip_horizontal=mirror_in_backend,
    )
    width, height = preview.output_size
    src_points = np.float32(preview.src_points)
    dst_points = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1],
        ]
    )
    final_transform = cv2.getPerspectiveTransform(src_points, dst_points)
    if mirror_in_backend:
        output_flip = np.float32(
            [
                [-1.0, 0.0, float(width - 1)],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        final_transform = output_flip @ final_transform
    inverse_transform = np.linalg.inv(final_transform)
    return inverse_transform, preview.source_size, mirror_in_backend


def _project_warp_point_to_raw(
    x_value: float | None,
    y_value: float | None,
    inverse_transform: np.ndarray,
    raw_size: tuple[int, int],
    mirror_in_backend: bool,
) -> tuple[float, float] | None:
    if x_value is None or y_value is None:
        return None
    raw_width, raw_height = raw_size
    if raw_width <= 0 or raw_height <= 0:
        return None
    source = np.asarray([[[float(x_value), float(y_value)]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(source, inverse_transform.astype(np.float32))
    if projected.size < 2:
        return None
    x_raw = float(projected[0, 0, 0])
    y_raw = float(projected[0, 0, 1])
    if mirror_in_backend:
        x_raw = float(raw_width - 1) - x_raw
    x_raw = max(0.0, min(float(raw_width - 1), x_raw))
    y_raw = max(0.0, min(float(raw_height - 1), y_raw))
    return x_raw, y_raw


def _marker_color_for_status(status: str) -> str:
    if status in {"left_only", "right_only", "left_only_manual", "right_only_manual"}:
        return "#f2a84b"
    return "#54b8ff"


def _clip_u8(value: float) -> int:
    return max(0, min(255, int(round(float(value)))))


def _hex_from_rgb(rgb_triplet: tuple[float, float, float]) -> str:
    r_value, g_value, b_value = rgb_triplet
    return f"#{_clip_u8(r_value):02x}{_clip_u8(g_value):02x}{_clip_u8(b_value):02x}"


def _rgb_luminance(rgb_triplet: tuple[float, float, float]) -> float:
    r_value, g_value, b_value = [float(channel) / 255.0 for channel in rgb_triplet]
    return (0.2126 * r_value) + (0.7152 * g_value) + (0.0722 * b_value)


def _adaptive_marker_style(
    image_bgr: np.ndarray,
    x_raw: float,
    y_raw: float,
    *,
    status: str,
) -> dict[str, Any]:
    image_height, image_width = image_bgr.shape[:2]
    x_center = max(0, min(image_width - 1, int(round(float(x_raw)))))
    y_center = max(0, min(image_height - 1, int(round(float(y_raw)))))
    radius = 10
    x0 = max(0, x_center - radius)
    x1 = min(image_width, x_center + radius + 1)
    y0 = max(0, y_center - radius)
    y1 = min(image_height, y_center + radius + 1)
    patch = image_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        patch = image_bgr[max(0, y_center - 1) : min(image_height, y_center + 2), max(0, x_center - 1) : min(image_width, x_center + 2)]
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    mean_rgb = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
    luminance = _rgb_luminance(mean_rgb)
    hue, saturation, _value = colorsys.rgb_to_hsv(*(channel / 255.0 for channel in mean_rgb))

    if status in {"left_only", "right_only", "left_only_manual", "right_only_manual"}:
        target_hue = 0.10
        target_saturation = 0.72
    else:
        if saturation < 0.14:
            warm_bias = (mean_rgb[0] - mean_rgb[2]) >= 10.0
            target_hue = 0.58 if warm_bias else 0.17
        else:
            target_hue = (hue + 0.46) % 1.0
        target_saturation = 0.58

    target_value = 0.84 if luminance < 0.44 else 0.58
    fill_rgb = colorsys.hsv_to_rgb(target_hue, target_saturation, target_value)
    fill_rgb_255 = tuple(channel * 255.0 for channel in fill_rgb)
    fill_luminance = _rgb_luminance(fill_rgb_255)
    text_color = "#0f1822" if fill_luminance > 0.63 else "#ffffff"
    text_stroke = "#f7f3e5" if fill_luminance < 0.48 else "#152131"
    return {
        "fill_color": _hex_from_rgb(fill_rgb_255),
        "fill_opacity": 0.82,
        "text_color": text_color,
        "text_stroke": text_stroke,
    }


def _build_side_markers(
    side: str,
    match_payload: dict[str, Any],
    dataset: dict[str, Any],
    inverse_transform: np.ndarray,
    raw_size: tuple[int, int],
    mirror_in_backend: bool,
    image_bgr: np.ndarray,
) -> list[dict[str, Any]]:
    side_key = _normalize_side(side)
    items_key = "left_items" if side_key == "151" else "right_items"
    rows = list(match_payload.get("rows") or [])
    raw_items = list(match_payload.get(items_key) or dataset.get("items") or [])
    item_lookup = {
        int(item.get("tube_idx")): item
        for item in raw_items
        if isinstance(item, dict) and item.get("tube_idx") is not None
    }
    markers: list[dict[str, Any]] = []
    row_tube_key = "tube_idx_151" if side_key == "151" else "tube_idx_152"

    for row in rows:
        if not isinstance(row, dict):
            continue
        tube_idx = row.get(row_tube_key)
        if tube_idx is None:
            continue
        item = item_lookup.get(int(tube_idx))
        if not isinstance(item, dict):
            continue
        source_measurement = item.get("source_measurement")
        if not isinstance(source_measurement, dict):
            continue
        x_value = None
        for key in ("x_start_raw_warp", "x_start_warp", "x_start_smooth_warp", "ref_x"):
            raw_number = source_measurement.get(key)
            if raw_number is not None:
                try:
                    x_value = float(raw_number)
                    break
                except (TypeError, ValueError):
                    continue
        y_value = None
        try:
            if source_measurement.get("y_center_warp") is not None:
                y_value = float(source_measurement.get("y_center_warp"))
        except (TypeError, ValueError):
            y_value = None
        projected = _project_warp_point_to_raw(x_value, y_value, inverse_transform, raw_size, mirror_in_backend)
        if projected is None:
            continue
        marker_style = _adaptive_marker_style(
            image_bgr,
            projected[0],
            projected[1],
            status=str(row.get("match_status") or ""),
        )
        markers.append(
            {
                "tube_number": int(row.get("tube_number") or len(markers) + 1),
                "tube_idx": int(tube_idx),
                "match_status": str(row.get("match_status") or ""),
                "x": round(projected[0], 3),
                "y": round(projected[1], 3),
                "display_color": _marker_color_for_status(str(row.get("match_status") or "")),
                "label_fill": str(marker_style.get("fill_color") or _marker_color_for_status(str(row.get("match_status") or ""))),
                "label_fill_opacity": float(marker_style.get("fill_opacity") or 0.82),
                "label_text": str(marker_style.get("text_color") or "#ffffff"),
                "label_text_stroke": str(marker_style.get("text_stroke") or "#152131"),
            }
        )
    markers.sort(key=lambda marker: int(marker.get("tube_number", 0)))
    return markers


def _format_length_ft(total_inches: float | None, denominator: int = 8) -> str:
    if total_inches is None:
        return "-"
    inches_value = float(total_inches)
    feet = int(math.floor(inches_value / 12.0))
    remainder = max(0.0, inches_value - (feet * 12.0))
    whole_inches = int(math.floor(remainder + 1e-9))
    fractional = max(0.0, remainder - whole_inches)
    numerator = int(round(fractional * denominator))
    if numerator >= denominator:
        whole_inches += 1
        numerator = 0
    if whole_inches >= 12:
        feet += whole_inches // 12
        whole_inches = whole_inches % 12
    if numerator == 0:
        return f"{feet} - {whole_inches}"
    common = math.gcd(numerator, denominator)
    numerator //= common
    reduced_den = denominator // common
    if whole_inches == 0:
        return f"{feet} - {numerator}/{reduced_den}"
    return f"{feet} - {whole_inches} & {numerator}/{reduced_den}"


def _build_table_rows(match_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_row in list(match_payload.get("rows") or []):
        if not isinstance(raw_row, dict):
            continue
        total_length = raw_row.get("total_length_in")
        try:
            total_length_value = None if total_length is None else float(total_length)
        except (TypeError, ValueError):
            total_length_value = None
        rows.append(
            {
                "tube_number": int(raw_row.get("tube_number") or len(rows) + 1),
                "match_status": str(raw_row.get("match_status") or ""),
                "length_in": total_length_value,
                "length_in_display": "-" if total_length_value is None else f"{total_length_value:.2f}",
                "length_ft_display": _format_length_ft(total_length_value),
                "tube_idx_151": raw_row.get("tube_idx_151"),
                "tube_idx_152": raw_row.get("tube_idx_152"),
                "observations": str(raw_row.get("observations") or ""),
            }
        )
    return rows


def _state_for_side(
    side: str,
    match_payload: dict[str, Any],
    dataset: dict[str, Any],
    image_path: Path,
    roi_path: Path,
    *,
    asset_url: str,
) -> dict[str, Any]:
    side_key = _normalize_side(side)
    raw_bytes = image_path.read_bytes()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise FileNotFoundError(f"Could not read the RAW image: {image_path}")
    raw_height, raw_width = image_bgr.shape[:2]
    roi_payload = load_rois(roi_path)
    transform_dir = _repo_root() / "artifacts" / "sorting_table_mvp" / f"cam{side_key}_{image_path.stem}"
    inverse_transform, raw_size, mirror_in_backend = _build_inverse_output_transform(image_path, roi_payload, transform_dir)
    markers = _build_side_markers(side_key, match_payload, dataset, inverse_transform, raw_size, mirror_in_backend, image_bgr)
    return {
        "side": side_key,
        "title": f"RAW {'LEFT' if side_key == '151' else 'RIGHT'}",
        "image_name": image_path.name,
        "url": str(asset_url),
        "mime_type": _image_mime_type(image_path),
        "width": int(raw_width),
        "height": int(raw_height),
        "markers": markers,
        "marker_count": len(markers),
        "mirror_in_backend": bool(mirror_in_backend),
        "resolved_image_path": str(image_path),
        "resolved_roi_path": str(roi_path),
        "asset_bytes": raw_bytes,
    }


def _build_initial_state(
    match_payload: dict[str, Any],
    left_state: dict[str, Any],
    right_state: dict[str, Any],
    *,
    current_run: dict[str, Any] | None,
    is_latest_view: bool,
) -> dict[str, Any]:
    rows = _build_table_rows(match_payload)
    summary = dict(match_payload.get("summary") or {})
    detection_source = ""
    pipe_end_yolo: dict[str, Any] = {}
    if current_run:
        detection_source = str(current_run.get("detection_source") or "")
        pipe_end_yolo = dict(current_run.get("pipe_end_yolo") or {})
    return {
            "title": "Sorting Table",
        "generated_at": str(match_payload.get("generated_at") or _iso_now()),
        "match_source": str(match_payload.get("_input_path") or "latest"),
        "history_url": "/history",
        "latest_url": "/",
        "capture_api_url": "/api/capture",
        "current_run": current_run,
        "is_latest_view": bool(is_latest_view),
        "summary": {
            "pipe_count": len(rows),
            "matched": int(summary.get("matched") or 0),
            "left_only": int(summary.get("left_only") or 0),
            "right_only": int(summary.get("right_only") or 0),
            "detection_source": detection_source,
            "pipe_end_yolo": pipe_end_yolo,
        },
        "rows": rows,
        "images": {
            "151": {key: value for key, value in left_state.items() if not key.startswith("asset_")},
            "152": {key: value for key, value in right_state.items() if not key.startswith("asset_")},
        },
    }


def _sorting_table_html(initial_state: dict[str, Any]) -> str:
    html_path = Path(__file__).with_name("sorting_table_mvp.html")
    html_template = html_path.read_text(encoding="utf-8")
    return html_template.replace("__INITIAL_STATE__", json.dumps(initial_state, ensure_ascii=False))


def _sorting_table_history_html(initial_state: dict[str, Any]) -> str:
    html_path = Path(__file__).with_name("sorting_table_history.html")
    html_template = html_path.read_text(encoding="utf-8")
    return html_template.replace("__HISTORY_STATE__", json.dumps(initial_state, ensure_ascii=False))


def _framing_calibrator_html(initial_state: dict[str, Any]) -> str:
    html_path = Path(__file__).with_name("framing_calibrator.html")
    html_template = html_path.read_text(encoding="utf-8")
    return html_template.replace("__CALIBRATOR_STATE__", json.dumps(initial_state, ensure_ascii=False))


def _homography_points_for_editor(roi_path: Path) -> dict[str, list[float]]:
    payload = load_rois(roi_path)
    points = payload.get("src_points_override")
    labels = ("TL", "TR", "BL", "BR")
    if not isinstance(points, list) or len(points) != 4:
        return {}
    result: dict[str, list[float]] = {}
    for label, point in zip(labels, points):
        try:
            result[label] = [float(point[0]), float(point[1])]
        except Exception:
            return {}
    return result


def _homography_editor_html(initial_state: dict[str, Any]) -> str:
    payload = json.dumps(initial_state, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Homography Editor</title>
<style>
*{{box-sizing:border-box}}
html,body{{margin:0;height:100%;font-family:Arial,Helvetica,sans-serif;background:#101216;color:#edf2f7}}
body{{display:grid;grid-template-columns:minmax(0,1fr)380px;grid-template-rows:64px minmax(0,1fr)}}
header{{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 14px;border-bottom:1px solid #2a3038;background:#171a20}}
h1{{margin:0;font-size:18px}}.actions{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
button,select,input{{border:1px solid #3a4450;border-radius:8px;background:#222832;color:#f5f7fa;padding:8px 10px;font-size:13px}}
button{{cursor:pointer}}button.primary{{border-color:#1f9d55;background:#1f7a46}}button.secondary{{background:#2d3440}}button.danger{{border-color:#9f3a3a;background:#6b2a2a}}button:disabled{{opacity:.45;cursor:not-allowed}}
main{{min-width:0;min-height:0;display:grid;grid-template-rows:minmax(0,1fr)34px;background:#0d0f13}}
#stage{{width:100%;height:100%;display:block;background:#0b0d10;touch-action:none;cursor:crosshair}}
#status{{padding:8px 12px;border-top:1px solid #2a3038;background:#171a20;color:#d7dee8;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
aside{{min-height:0;overflow:auto;border-left:1px solid #2a3038;background:#151920;padding:12px}}
.panel{{border:1px solid #2c3440;border-radius:10px;background:#1b2028;padding:10px;margin-bottom:12px}}
.panel h2{{margin:0 0 8px;font-size:13px;text-transform:uppercase;color:#9fb0c2;letter-spacing:.06em}}
#preview{{width:100%;height:auto;display:block;border:1px solid #303844;background:#0d0f13}}
.field{{display:grid;gap:6px;margin-bottom:10px}}label{{font-size:12px;color:#b7c2ce}}select,input{{width:100%}}
.point-list{{display:grid;gap:8px}}.point-row{{display:grid;grid-template-columns:34px minmax(0,1fr);gap:8px;align-items:center;padding:8px;border-radius:6px;background:#222832}}
.badge{{width:26px;height:26px;display:grid;place-items:center;border-radius:50%;font-weight:700;color:#0d0f13;font-size:12px}}
.coords{{min-width:0;font-size:12px;color:#d7dee8;line-height:1.35;overflow-wrap:anywhere}}.coords strong{{display:block;color:#fff;font-size:13px}}
.help,.meta{{margin:0;color:#b7c2ce;font-size:12px;line-height:1.45;overflow-wrap:anywhere}}
@media(max-width:980px){{body{{grid-template-columns:1fr;grid-template-rows:auto minmax(62vh,1fr)auto}}aside{{border-left:0;border-top:1px solid #2a3038;max-height:38vh}}header{{align-items:flex-start;flex-direction:column}}}}
</style>
</head>
<body>
<header>
  <h1>Homography Editor</h1>
  <div class="actions">
    <button id="fitBtn">Fit</button>
    <button id="deleteBtn">Delete last</button>
    <button id="clearBtn">Clear</button>
    <button id="defaultBtn" class="secondary">Use selected as default</button>
    <button id="saveBtn" class="primary" disabled>Save new profile + default</button>
  </div>
</header>
<main><canvas id="stage"></canvas><div id="status">Loading image...</div></main>
<aside>
  <section class="panel">
    <h2>Run</h2>
    <p class="meta" id="runMeta"></p>
  </section>
  <section class="panel">
    <h2>Profile</h2>
    <div class="field"><label for="profileSelect">Load previous TOML</label><select id="profileSelect"></select></div>
    <div class="field"><label for="profileName">Optional name for new save</label><input id="profileName" placeholder="morning_cam152, fixed_shift..."></div>
  </section>
  <section class="panel"><h2>Warp preview</h2><canvas id="preview" width="320" height="460"></canvas></section>
  <section class="panel"><h2>Points</h2><div id="pointList" class="point-list"></div></section>
  <section class="panel"><h2>Controls</h2><p class="help">Left click places the next point. Drag a point to move it. Mouse wheel zooms. Shift, middle click, or right click pans. Enter saves. Delete removes last point.</p></section>
</aside>
<script>
const INITIAL={payload};
const LABELS=["TL","TR","BL","BR"],NAMES={{TL:"P1 top-left",TR:"P2 top-right",BL:"P3 bottom-left",BR:"P4 bottom-right"}},COLORS={{TL:"#00ff44",TR:"#ffcc00",BL:"#ff4444",BR:"#44aaff"}};
const stage=document.getElementById("stage"),ctx=stage.getContext("2d"),preview=document.getElementById("preview"),pctx=preview.getContext("2d");
const statusEl=document.getElementById("status"),saveBtn=document.getElementById("saveBtn"),pointList=document.getElementById("pointList"),profileSelect=document.getElementById("profileSelect");
const img=new Image(),srcCanvas=document.createElement("canvas"),srcCtx=srcCanvas.getContext("2d",{{willReadFrequently:true}});
let srcPixels=null,drag=null,previewDirty=false;
const state={{pts:{{}},nextIdx:0,zoom:1,offsetX:0,offsetY:0,fitted:false,history:[]}};
for(const l of LABELS){{const p=INITIAL.existing[l];if(Array.isArray(p)&&p.length>=2)state.pts[l]={{x:Number(p[0]),y:Number(p[1])}};}}
state.nextIdx=nextIndex();
document.getElementById("runMeta").innerHTML=`Camera: <strong>cam${{INITIAL.camera}}</strong><br>Run: <strong>${{INITIAL.run_id||"latest"}}</strong><br>Image: ${{INITIAL.image.name}}<br>Selected TOML: ${{INITIAL.selected_profile_path}}`;
profileSelect.innerHTML=(INITIAL.profiles||[]).map(p=>`<option value="${{p.profile_id}}" ${{p.profile_id===INITIAL.selected_profile_id?"selected":""}}>${{p.is_current?"* ":""}}${{p.label}} - ${{p.image_name||""}}</option>`).join("");
profileSelect.onchange=()=>{{const params=new URLSearchParams(location.search);params.set("camera",INITIAL.camera);if(INITIAL.run_id)params.set("run_id",INITIAL.run_id);params.set("profile_id",profileSelect.value);location.href="/homography-editor?"+params.toString();}};
function allPlaced(){{return LABELS.every(l=>state.pts[l]);}}
function nextIndex(){{const i=LABELS.findIndex(l=>!state.pts[l]);return i>=0?i:state.nextIdx%LABELS.length;}}
function clamp(p){{return{{x:Math.max(0,Math.min(img.naturalWidth,Number(p.x))),y:Math.max(0,Math.min(img.naturalHeight,Number(p.y)))}};}}
function canvasPoint(e){{const r=stage.getBoundingClientRect();return{{x:e.clientX-r.left,y:e.clientY-r.top}};}}
function i2c(p){{return{{x:p.x*state.zoom+state.offsetX,y:p.y*state.zoom+state.offsetY}};}}
function c2i(p){{return{{x:(p.x-state.offsetX)/state.zoom,y:(p.y-state.offsetY)/state.zoom}};}}
function setStatus(t,err=false){{statusEl.textContent=t;statusEl.style.color=err?"#ff8585":"#d7dee8";}}
function fitImage(){{if(!img.naturalWidth||!stage.width||!stage.height)return;const s=Math.min(stage.width/img.naturalWidth,stage.height/img.naturalHeight,1);state.zoom=Math.max(.02,s);state.offsetX=(stage.width-img.naturalWidth*state.zoom)/2;state.offsetY=(stage.height-img.naturalHeight*state.zoom)/2;state.fitted=true;draw();}}
function resizeStage(){{const r=stage.getBoundingClientRect();stage.width=Math.max(1,Math.round(r.width));stage.height=Math.max(1,Math.round(r.height));state.fitted?draw():fitImage();}}
function hitPoint(cp){{let best=null,bd=18;for(const l of LABELS){{const p=state.pts[l];if(!p)continue;const d=i2c(p),dist=Math.hypot(cp.x-d.x,cp.y-d.y);if(dist<bd){{bd=dist;best=l;}}}}return best;}}
function placePoint(l,p){{state.pts[l]=clamp(p);state.history.push(l);state.nextIdx=(LABELS.indexOf(l)+1)%LABELS.length;if(!allPlaced())state.nextIdx=nextIndex();markPreviewDirty();draw();}}
function deleteLast(){{let l=state.history.pop();if(!l||!state.pts[l])l=LABELS[(state.nextIdx+LABELS.length-1)%LABELS.length];if(l){{delete state.pts[l];state.nextIdx=LABELS.indexOf(l);}}markPreviewDirty();draw();}}
function clearPoints(){{state.pts={{}};state.history=[];state.nextIdx=0;markPreviewDirty();draw();}}
function drawPointList(){{pointList.innerHTML="";for(const l of LABELS){{const row=document.createElement("div"),b=document.createElement("div"),c=document.createElement("div"),p=state.pts[l];row.className="point-row";b.className="badge";b.style.background=COLORS[l];b.textContent=l;c.className="coords";c.innerHTML=p?`<strong>${{NAMES[l]}}</strong>x=${{p.x.toFixed(1)}}, y=${{p.y.toFixed(1)}}`:`<strong>${{NAMES[l]}}</strong>not set`;row.append(b,c);pointList.appendChild(row);}}}}
function draw(){{ctx.clearRect(0,0,stage.width,stage.height);ctx.fillStyle="#0b0d10";ctx.fillRect(0,0,stage.width,stage.height);if(img.complete&&img.naturalWidth)ctx.drawImage(img,state.offsetX,state.offsetY,img.naturalWidth*state.zoom,img.naturalHeight*state.zoom);if(allPlaced()){{const tl=i2c(state.pts.TL),tr=i2c(state.pts.TR),bl=i2c(state.pts.BL),br=i2c(state.pts.BR);ctx.save();ctx.strokeStyle="rgba(255,255,255,.78)";ctx.lineWidth=2;ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(tl.x,tl.y);ctx.lineTo(tr.x,tr.y);ctx.lineTo(br.x,br.y);ctx.lineTo(bl.x,bl.y);ctx.closePath();ctx.stroke();ctx.restore();}}for(const l of LABELS){{const p=state.pts[l];if(!p)continue;const d=i2c(p);ctx.save();ctx.strokeStyle=COLORS[l];ctx.fillStyle=COLORS[l];ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(d.x-10,d.y);ctx.lineTo(d.x+10,d.y);ctx.moveTo(d.x,d.y-10);ctx.lineTo(d.x,d.y+10);ctx.stroke();ctx.beginPath();ctx.arc(d.x,d.y,5,0,Math.PI*2);ctx.stroke();ctx.font="700 13px Arial,Helvetica,sans-serif";ctx.fillText(NAMES[l],d.x+12,d.y-10);ctx.restore();}}saveBtn.disabled=!allPlaced();drawPointList();if(allPlaced())setStatus("4/4 points ready. Save to create a TOML profile and make it default.");else{{const l=LABELS[state.nextIdx];setStatus(`Next: ${{NAMES[l]}} (${{Object.keys(state.pts).length}}/4)`);}}}}
function markPreviewDirty(){{if(previewDirty)return;previewDirty=true;requestAnimationFrame(()=>{{previewDirty=false;drawPreview();}});}}
function drawPreview(){{pctx.clearRect(0,0,preview.width,preview.height);pctx.fillStyle="#0d0f13";pctx.fillRect(0,0,preview.width,preview.height);if(!allPlaced()||!srcPixels){{pctx.fillStyle="#9fb0c2";pctx.font="14px Arial";pctx.textAlign="center";pctx.fillText(`${{Object.keys(state.pts).length}}/4 points`,preview.width/2,preview.height/2);pctx.textAlign="start";return;}}const dst=[{{x:0,y:0}},{{x:preview.width-1,y:0}},{{x:0,y:preview.height-1}},{{x:preview.width-1,y:preview.height-1}}],src=[state.pts.TL,state.pts.TR,state.pts.BL,state.pts.BR],h=solveHomography(dst,src);if(!h)return;const out=pctx.createImageData(preview.width,preview.height),sd=srcPixels.data,sw=srcPixels.width,sh=srcPixels.height;for(let y=0;y<preview.height;y++){{for(let x=0;x<preview.width;x++){{const den=h[6]*x+h[7]*y+1,sx=(h[0]*x+h[1]*y+h[2])/den,sy=(h[3]*x+h[4]*y+h[5])/den,oi=(y*preview.width+x)*4,ix=Math.round(sx),iy=Math.round(sy);if(ix>=0&&ix<sw&&iy>=0&&iy<sh){{const si=(iy*sw+ix)*4;out.data[oi]=sd[si];out.data[oi+1]=sd[si+1];out.data[oi+2]=sd[si+2];out.data[oi+3]=255;}}}}pctx.putImageData(out,0,0);}}
function solveHomography(from,to){{const a=[],b=[];for(let i=0;i<4;i++){{const x=from[i].x,y=from[i].y,u=to[i].x,v=to[i].y;a.push([x,y,1,0,0,0,-u*x,-u*y]);b.push(u);a.push([0,0,0,x,y,1,-v*x,-v*y]);b.push(v);}}const s=gauss(a,b);return s?[s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],1]:null;}}
function gauss(a,b){{const n=b.length;for(let c=0;c<n;c++){{let p=c;for(let r=c+1;r<n;r++)if(Math.abs(a[r][c])>Math.abs(a[p][c]))p=r;if(Math.abs(a[p][c])<1e-10)return null;if(p!==c){{[a[p],a[c]]=[a[c],a[p]];[b[p],b[c]]=[b[c],b[p]];}}const d=a[c][c];for(let j=c;j<n;j++)a[c][j]/=d;b[c]/=d;for(let r=0;r<n;r++){{if(r===c)continue;const f=a[r][c];for(let j=c;j<n;j++)a[r][j]-=f*a[c][j];b[r]-=f*b[c];}}}}return b;}}
stage.addEventListener("pointerdown",e=>{{e.preventDefault();stage.setPointerCapture(e.pointerId);const cp=canvasPoint(e);if(e.button===1||e.button===2||e.shiftKey||e.altKey){{drag={{type:"pan",sx:e.clientX,sy:e.clientY,ox:state.offsetX,oy:state.offsetY}};return;}}if(e.button!==0)return;const hit=hitPoint(cp);if(hit){{drag={{type:"point",label:hit}};return;}}const ip=c2i(cp);if(ip.x>=0&&ip.x<=img.naturalWidth&&ip.y>=0&&ip.y<=img.naturalHeight)placePoint(LABELS[state.nextIdx],ip);}});
stage.addEventListener("pointermove",e=>{{if(!drag)return;e.preventDefault();if(drag.type==="pan"){{state.offsetX=drag.ox+e.clientX-drag.sx;state.offsetY=drag.oy+e.clientY-drag.sy;draw();return;}}state.pts[drag.label]=clamp(c2i(canvasPoint(e)));markPreviewDirty();draw();}});
function stopDrag(e){{if(stage.hasPointerCapture(e.pointerId))stage.releasePointerCapture(e.pointerId);drag=null;}}stage.addEventListener("pointerup",stopDrag);stage.addEventListener("pointercancel",stopDrag);stage.addEventListener("contextmenu",e=>e.preventDefault());
stage.addEventListener("wheel",e=>{{e.preventDefault();const cp=canvasPoint(e),before=c2i(cp),f=e.deltaY<0?1.12:1/1.12;state.zoom=Math.max(.02,Math.min(30,state.zoom*f));state.offsetX=cp.x-before.x*state.zoom;state.offsetY=cp.y-before.y*state.zoom;draw();}},{{passive:false}});
function pointsPayload(){{const pts={{}};for(const l of LABELS)pts[l]=[state.pts[l].x,state.pts[l].y];return pts;}}
async function postJson(url,payload){{const r=await fetch(url,{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify(payload)}});const data=await r.json().catch(()=>({{}}));if(!r.ok||data.ok===false)throw new Error(data.error||"Request failed.");return data;}}
async function saveProfile(){{if(!allPlaced())return;saveBtn.disabled=true;setStatus("Saving homography profile...");try{{const data=await postJson("/api/homography-profiles",{{camera:INITIAL.camera,run_id:INITIAL.run_id,base_profile_id:INITIAL.selected_profile_id,name:document.getElementById("profileName").value,points:pointsPayload(),set_as_default:true}});setStatus(`Saved ${{data.profile_id}} and set as default.`);setTimeout(()=>location.href=`/homography-editor?camera=${{INITIAL.camera}}&run_id=${{INITIAL.run_id||""}}&profile_id=${{data.profile_id}}`,800);}}catch(e){{saveBtn.disabled=false;setStatus(e.message||String(e),true);}}}}
async function setSelectedDefault(){{setStatus("Setting selected profile as default...");try{{const data=await postJson("/api/homography-profiles/default",{{camera:INITIAL.camera,profile_id:profileSelect.value}});setStatus(`Default updated from ${{data.profile_id}}.`);}}catch(e){{setStatus(e.message||String(e),true);}}}}
document.getElementById("fitBtn").onclick=fitImage;document.getElementById("deleteBtn").onclick=deleteLast;document.getElementById("clearBtn").onclick=clearPoints;saveBtn.onclick=saveProfile;document.getElementById("defaultBtn").onclick=setSelectedDefault;
window.addEventListener("keydown",e=>{{if(e.key==="Enter"&&allPlaced())saveProfile();if(e.key==="Delete"||e.key==="Backspace"){{e.preventDefault();deleteLast();}}}});
window.addEventListener("resize",resizeStage);
img.onload=()=>{{srcCanvas.width=img.naturalWidth;srcCanvas.height=img.naturalHeight;srcCtx.drawImage(img,0,0);srcPixels=srcCtx.getImageData(0,0,img.naturalWidth,img.naturalHeight);resizeStage();markPreviewDirty();}};
img.onerror=()=>setStatus("Could not load image.",true);img.src=INITIAL.image.url;
</script>
</body>
</html>"""


def _baseline_image_path_for_side(side: str) -> Path:
    side_key = _normalize_side(side)
    filename = "cam_151_202604022.jpeg" if side_key == "151" else "cam_152_202604022.jpeg"
    path = _repo_root() / "test_images" / filename
    if not path.exists():
        raise FileNotFoundError(f"The April 22 base image for cam{side_key} does not exist: {path}")
    return path


def _load_image_dimensions(image_path: Path) -> tuple[int, int]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise FileNotFoundError(f"Could not read the image: {image_path}")
    height, width = image_bgr.shape[:2]
    return int(width), int(height)


def _camera_overlay_path_from_manifest(manifest: dict[str, Any], side: str) -> tuple[Path, str]:
    side_key = _normalize_side(side)
    camera_key = f"cam{side_key}"
    camera_entry = dict((manifest.get("cameras") or {}).get(camera_key) or {})
    for field_name, source_kind in (
        ("wide_source_image_path", "wide_source"),
        ("image_path", "processed_capture"),
    ):
        candidate = _resolve_existing_path(camera_entry.get(field_name))
        if candidate is not None:
            return candidate, source_kind
    raise FileNotFoundError(f"Run {manifest.get('run_id')!r} does not have a usable image for cam{side_key}.")


def _calibrator_camera_state(
    manifest: dict[str, Any],
    side: str,
    *,
    asset_namespace: str,
) -> tuple[dict[str, Any], dict[str, tuple[bytes, str]]]:
    side_key = _normalize_side(side)
    camera_key = f"cam{side_key}"
    camera_entry = dict((manifest.get("cameras") or {}).get(camera_key) or {})
    baseline_path = _baseline_image_path_for_side(side_key)
    overlay_path, overlay_source_kind = _camera_overlay_path_from_manifest(manifest, side_key)
    baseline_width, baseline_height = _load_image_dimensions(baseline_path)
    overlay_width, overlay_height = _load_image_dimensions(overlay_path)
    baseline_url = f"/asset/{asset_namespace}/baseline_cam{side_key}{baseline_path.suffix.lower()}"
    overlay_url = f"/asset/{asset_namespace}/overlay_cam{side_key}{overlay_path.suffix.lower()}"
    digital_framing = dict(camera_entry.get("digital_framing") or {})
    initial_zoom_factor = max(1.0, float(digital_framing.get("zoom_factor") or 1.0))
    initial_offset_x = float(digital_framing.get("offset_x") or 0.0)
    initial_offset_y = float(digital_framing.get("offset_y") or 0.0)
    initial_opacity = 0.55
    return (
        {
            "side": side_key,
            "label": f"cam{side_key}",
            "baseline": {
                "image_name": baseline_path.name,
                "url": baseline_url,
                "width": baseline_width,
                "height": baseline_height,
                "resolved_path": str(baseline_path),
            },
            "overlay": {
                "image_name": overlay_path.name,
                "url": overlay_url,
                "width": overlay_width,
                "height": overlay_height,
                "resolved_path": str(overlay_path),
                "source_kind": overlay_source_kind,
            },
            "initial_controls": {
                "zoom_factor": round(initial_zoom_factor, 4),
                "offset_x": round(initial_offset_x, 6),
                "offset_y": round(initial_offset_y, 6),
                "opacity": initial_opacity,
            },
            "current_capture_ptz": dict(camera_entry.get("ptz") or {}),
            "current_digital_framing": digital_framing,
        },
        {
            baseline_url: (baseline_path.read_bytes(), _image_mime_type(baseline_path)),
            overlay_url: (overlay_path.read_bytes(), _image_mime_type(overlay_path)),
        },
    )


def start_sorting_table_mvp_server(
    match_source: str | Path | dict[str, Any] | None = None,
    *,
    cam151_input: str | Path | dict[str, Any] | None = None,
    cam152_input: str | Path | dict[str, Any] | None = None,
    open_browser: bool = True,
    on_ready: Any | None = None,
    port: int | None = None,
) -> SortingTableMvpServerHandle:
    runtime_lock = threading.Lock()
    capture_lock = threading.Lock()
    asset_cache: dict[str, tuple[bytes, str]] = {}

    def _build_runtime_snapshot(*, requested_run_id: str | None = None, requested_artifact_name: str | None = None) -> dict[str, Any]:
        latest_manifest = load_latest_capture_run_manifest() if match_source is None else None
        current_run_manifest: dict[str, Any] | None = None
        current_run_state_override: dict[str, Any] | None = None
        if requested_run_id:
            current_run_manifest = load_capture_run_manifest(requested_run_id)
            current_match_source = _resolve_manifest_match_source(current_run_manifest)
        elif requested_artifact_name:
            current_match_source = _resolve_legacy_match_artifact(requested_artifact_name)
            current_run_state_override = _history_entry_from_legacy_match(Path(current_match_source))
        else:
            current_match_source = match_source if match_source is not None else find_latest_match_result()
            current_run_manifest = latest_manifest

        match_payload = _load_match_result(current_match_source)
        left_dataset = _resolve_dataset_from_match(match_payload, "151", source_override=cam151_input)
        right_dataset = _resolve_dataset_from_match(match_payload, "152", source_override=cam152_input)
        left_image_path = _resolve_raw_image_path(left_dataset, match_payload, "151")
        right_image_path = _resolve_raw_image_path(right_dataset, match_payload, "152")
        left_roi_path = _resolve_roi_path(left_dataset, match_payload, "151")
        right_roi_path = _resolve_roi_path(right_dataset, match_payload, "152")
        latest_run_id = str((latest_manifest or {}).get("run_id") or "").strip() or None
        current_run_id = str((current_run_manifest or {}).get("run_id") or "").strip()
        asset_namespace = _asset_token(
            current_run_id or f"{left_image_path.stem}_{right_image_path.stem}_{match_payload.get('generated_at')}",
            fallback="sorting_table_latest",
        )
        left_asset_url = f"/asset/{asset_namespace}/cam151{left_image_path.suffix.lower()}"
        right_asset_url = f"/asset/{asset_namespace}/cam152{right_image_path.suffix.lower()}"
        left_state = _state_for_side(
            "151",
            match_payload,
            left_dataset,
            left_image_path,
            left_roi_path,
            asset_url=left_asset_url,
        )
        right_state = _state_for_side(
            "152",
            match_payload,
            right_dataset,
            right_image_path,
            right_roi_path,
            asset_url=right_asset_url,
        )
        asset_map = {
            str(left_state["url"]): (left_state["asset_bytes"], left_state["mime_type"]),
            str(right_state["url"]): (right_state["asset_bytes"], right_state["mime_type"]),
        }
        current_run_state = current_run_state_override or _run_state_from_manifest(current_run_manifest, latest_run_id=latest_run_id)
        initial_state = _build_initial_state(
            match_payload,
            left_state,
            right_state,
            current_run=current_run_state,
            is_latest_view=bool(
                requested_run_id is None
                or (current_run_state and current_run_state.get("is_latest"))
            ),
        )
        html = _sorting_table_html(initial_state)
        return {
            "asset_map": asset_map,
            "html_bytes": html.encode("utf-8"),
        }

    def _build_history_state() -> dict[str, Any]:
        latest_manifest = load_latest_capture_run_manifest()
        latest_run_id = str((latest_manifest or {}).get("run_id") or "").strip() or None
        entries = [_history_entry_from_manifest(manifest, latest_run_id=latest_run_id) for manifest in list_capture_run_manifests()]
        entries.extend(_list_legacy_history_entries())
        entries.sort(key=lambda item: str(item.get("captured_at") or item.get("run_id") or ""), reverse=True)
        return {
            "title": "Download History",
            "history_url": "/history",
            "latest_url": "/",
            "capture_api_url": "/api/capture",
            "delete_api_url": "/api/history",
            "runs": entries,
            "latest_run_id": latest_run_id,
        }

    def _build_framing_calibrator_snapshot(
        *,
        requested_run_id: str | None = None,
        requested_camera: str | None = None,
    ) -> dict[str, Any]:
        manifest = load_capture_run_manifest(requested_run_id) if requested_run_id else load_latest_capture_run_manifest()
        if manifest is None:
            raise FileNotFoundError("There are no captured runs available for calibration.")
        run_id = str(manifest.get("run_id") or "").strip()
        latest_manifest = load_latest_capture_run_manifest()
        latest_run_id = str((latest_manifest or {}).get("run_id") or "").strip() or None
        camera_key = _normalize_side(requested_camera or "152")
        asset_namespace = _asset_token(f"framing_{run_id}", fallback="framing_calibrator")
        left_state, left_assets = _calibrator_camera_state(manifest, "151", asset_namespace=asset_namespace)
        right_state, right_assets = _calibrator_camera_state(manifest, "152", asset_namespace=asset_namespace)
        available_runs = [
            {
                "run_id": str(item.get("run_id") or ""),
                "captured_at": str(item.get("captured_at") or ""),
                "status": str(item.get("status") or ""),
            }
            for item in list_capture_run_manifests()
        ]
        initial_state = {
            "title": "Digital Framing Calibrator",
            "generated_at": _iso_now(),
            "latest_url": "/",
            "history_url": "/history",
            "current_run_id": run_id,
            "latest_run_id": latest_run_id,
            "current_camera": camera_key,
            "runs": available_runs,
            "cameras": {
                "151": left_state,
                "152": right_state,
            },
        }
        html = _framing_calibrator_html(initial_state)
        asset_map: dict[str, tuple[bytes, str]] = {}
        asset_map.update(left_assets)
        asset_map.update(right_assets)
        return {
            "asset_map": asset_map,
            "html_bytes": html.encode("utf-8"),
        }

    def _manifest_image_path_for_side(manifest: dict[str, Any], side: str) -> Path:
        side_key = _normalize_side(side)
        camera_key = f"cam{side_key}"
        camera_entry = dict((manifest.get("cameras") or {}).get(camera_key) or {})
        image_path = _resolve_existing_path(camera_entry.get("image_path"))
        if image_path is None:
            raise FileNotFoundError(f"Run {manifest.get('run_id')!r} does not have a usable raw image for cam{side_key}.")
        return image_path

    def _build_homography_editor_snapshot(
        *,
        requested_run_id: str | None = None,
        requested_camera: str | None = None,
        requested_profile_id: str | None = None,
    ) -> dict[str, Any]:
        manifest = load_capture_run_manifest(requested_run_id) if requested_run_id else load_latest_capture_run_manifest()
        if manifest is None:
            raise FileNotFoundError("There are no captured runs available for homography editing.")
        run_id = str(manifest.get("run_id") or "").strip()
        side_key = _normalize_side(requested_camera or "152")
        profile_id = str(requested_profile_id or "current_default").strip() or "current_default"
        image_path = _manifest_image_path_for_side(manifest, side_key)
        roi_path = resolve_homography_profile_path(side_key, profile_id)
        width, height = _load_image_dimensions(image_path)
        asset_namespace = _asset_token(f"homography_{run_id}_{side_key}", fallback="homography_editor")
        image_url = f"/asset/{asset_namespace}/cam{side_key}{image_path.suffix.lower()}"
        initial_state = {
            "title": "Homography Editor",
            "generated_at": _iso_now(),
            "latest_url": "/",
            "history_url": "/history",
            "camera": side_key,
            "run_id": run_id,
            "selected_profile_id": profile_id,
            "selected_profile_path": str(roi_path),
            "profiles": list_homography_profiles(side_key),
            "existing": _homography_points_for_editor(roi_path),
            "image": {
                "name": image_path.name,
                "url": image_url,
                "width": width,
                "height": height,
                "resolved_path": str(image_path),
            },
        }
        html = _homography_editor_html(initial_state)
        return {
            "asset_map": {image_url: (image_path.read_bytes(), _image_mime_type(image_path))},
            "html_bytes": html.encode("utf-8"),
        }

    runtime_box: dict[str, Any] = _build_runtime_snapshot()
    asset_cache.update(runtime_box.get("asset_map") or {})

    class SortingTableHandler(BaseHTTPRequestHandler):
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
            self._send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def _read_json_body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                length = 0
            raw = self.rfile.read(max(0, length)).decode("utf-8") if length else ""
            payload = json.loads(raw) if raw.strip() else {}
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
            return payload

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/homography-editor":
                query = parse_qs(parsed.query)
                requested_run_id = str((query.get("run_id") or [""])[0]).strip() or None
                requested_camera = str((query.get("camera") or [""])[0]).strip() or None
                requested_profile_id = str((query.get("profile_id") or [""])[0]).strip() or None
                try:
                    snapshot = _build_homography_editor_snapshot(
                        requested_run_id=requested_run_id,
                        requested_camera=requested_camera,
                        requested_profile_id=requested_profile_id,
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)
                    return
                with runtime_lock:
                    asset_cache.update(snapshot.get("asset_map") or {})
                self._send_bytes(snapshot["html_bytes"], "text/html; charset=utf-8")
                return
            if parsed.path == "/framing-calibrator":
                requested_run_id = str((parse_qs(parsed.query).get("run_id") or [""])[0]).strip() or None
                requested_camera = str((parse_qs(parsed.query).get("camera") or [""])[0]).strip() or None
                try:
                    snapshot = _build_framing_calibrator_snapshot(
                        requested_run_id=requested_run_id,
                        requested_camera=requested_camera,
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)
                    return
                with runtime_lock:
                    asset_cache.update(snapshot.get("asset_map") or {})
                self._send_bytes(snapshot["html_bytes"], "text/html; charset=utf-8")
                return
            if parsed.path == "/history":
                try:
                    html = _sorting_table_history_html(_build_history_state())
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)
                    return
                self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/history":
                try:
                    self._send_json(_build_history_state())
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)
                return
            if parsed.path in {"/", "/index.html"}:
                requested_run_id = str((parse_qs(parsed.query).get("run_id") or [""])[0]).strip() or None
                requested_artifact_name = str((parse_qs(parsed.query).get("artifact") or [""])[0]).strip() or None
                with runtime_lock:
                    try:
                        snapshot = _build_runtime_snapshot(
                            requested_run_id=requested_run_id,
                            requested_artifact_name=requested_artifact_name,
                        )
                    except Exception as exc:
                        self._send_json({"error": str(exc)}, status=500)
                        return
                    runtime_box.update(snapshot)
                    asset_cache.update(snapshot.get("asset_map") or {})
                    html_bytes = snapshot["html_bytes"]
                self._send_bytes(html_bytes, "text/html; charset=utf-8")
                return
            with runtime_lock:
                asset = asset_cache.get(parsed.path)
            if asset is not None:
                self._send_bytes(asset[0], asset[1])
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/homography-profiles":
                try:
                    payload = self._read_json_body()
                    side_key = _normalize_side(payload.get("camera"))
                    run_id = str(payload.get("run_id") or "").strip() or None
                    manifest = load_capture_run_manifest(run_id) if run_id else load_latest_capture_run_manifest()
                    if manifest is None:
                        raise FileNotFoundError("There are no captured runs available for homography editing.")
                    image_path = _manifest_image_path_for_side(manifest, side_key)
                    result = save_homography_profile(
                        side=side_key,
                        image_path=image_path,
                        src_points=payload.get("points"),
                        base_profile_id=str(payload.get("base_profile_id") or payload.get("profile_id") or "current_default").strip(),
                        name=str(payload.get("name") or "").strip(),
                        set_as_default=bool(payload.get("set_as_default", True)),
                    )
                    self._send_json({"ok": True, **result, "profiles": list_homography_profiles(side_key)})
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            if parsed.path == "/api/homography-profiles/default":
                try:
                    payload = self._read_json_body()
                    side_key = _normalize_side(payload.get("camera"))
                    profile_id = str(payload.get("profile_id") or "").strip()
                    if not profile_id:
                        raise ValueError("Missing profile_id.")
                    result = set_default_homography_profile(side_key, profile_id)
                    self._send_json({"ok": True, **result, "profiles": list_homography_profiles(side_key)})
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            if parsed.path != "/api/capture":
                self._send_json({"error": "not found"}, status=404)
                return
            if not capture_lock.acquire(blocking=False):
                self._send_json({"error": "A capture/processing job is already in progress."}, status=409)
                return
            try:
                manifest = capture_and_process_pair()
                with runtime_lock:
                    if match_source is None:
                        snapshot = _build_runtime_snapshot()
                        runtime_box.update(snapshot)
                        asset_cache.update(snapshot.get("asset_map") or {})
                current_run = _run_state_from_manifest(manifest, latest_run_id=str(manifest.get("run_id") or ""))
                self._send_json(
                    {
                        "ok": True,
                        "run_id": str(manifest.get("run_id") or ""),
                        "run_url": f"/?run_id={manifest.get('run_id')}",
                        "history_url": "/history",
                        "latest_url": "/",
                        "current_run": current_run,
                        "summary": dict((((manifest.get("processing") or {}).get("summary")) or {})),
                    }
                )
            except CaptureRunError as exc:
                self._send_json(
                    {
                        "error": str(exc),
                        "run_id": exc.run_id,
                        "manifest_path": None if exc.manifest_path is None else str(exc.manifest_path),
                        "history_url": "/history",
                    },
                    status=500,
                )
            except Exception as exc:
                self._send_json({"error": str(exc), "history_url": "/history"}, status=500)
            finally:
                capture_lock.release()

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            prefix = "/api/history/"
            if not parsed.path.startswith(prefix):
                self._send_json({"error": "not found"}, status=404)
                return
            run_id = parsed.path[len(prefix):].strip("/")
            if not run_id:
                self._send_json({"error": "Missing run_id."}, status=400)
                return
            try:
                result = delete_capture_run(run_id)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
                return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
                return
            self._send_json({"ok": True, **result})

    bind_port = int(port or 0)
    server = ThreadingHTTPServer(("127.0.0.1", bind_port), SortingTableHandler)
    host, bound_port = server.server_address
    url = f"http://{host}:{bound_port}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Sorting Table MVP ready: {url}")
    _display_jupyter_link(url)
    if on_ready is not None:
        try:
            on_ready(url)
        except Exception as exc:
            print(f"Warning: on_ready callback failed: {exc}")
    if open_browser:
        try:
            webbrowser.open(url, new=1)
        except Exception:
            pass
    return SortingTableMvpServerHandle(url=url, server=server, thread=thread)
