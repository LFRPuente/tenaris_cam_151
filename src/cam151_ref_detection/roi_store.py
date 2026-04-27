from __future__ import annotations

import json
import tomllib
from datetime import datetime
from pathlib import Path

import cv2


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def default_payload() -> dict:
    return {
        "image_name": None,
        "updated_at": None,
        "overview_path": None,
        "src_points_override": None,
        "dst_rect_override": None,
        "rois": [],
        "points": [],
        "lines": [],
    }


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _normalize_src_points_override(value: object) -> list[list[float]] | None:
    if not value:
        return None
    if not isinstance(value, list) or len(value) != 4:
        return None

    normalized: list[list[float]] = []
    for point in value:
        try:
            if isinstance(point, dict):
                x = float(point.get("x"))
                y = float(point.get("y"))
            else:
                x = float(point[0])
                y = float(point[1])
        except (TypeError, ValueError, IndexError):
            return None
        normalized.append([x, y])
    return normalized


def _normalize_dst_rect_override(value: object) -> list[float] | None:
    if not value:
        return None
    try:
        if isinstance(value, dict):
            rect = [value.get("x0"), value.get("y0"), value.get("x1"), value.get("y1")]
        else:
            rect = list(value)
        if len(rect) != 4:
            return None
        normalized = [float(item) for item in rect]
    except (TypeError, ValueError):
        return None
    return normalized


def _normalize_loaded_payload(payload: dict) -> dict:
    normalized = default_payload()
    normalized["image_name"] = payload.get("image_name") or payload.get("image_path")
    normalized["updated_at"] = payload.get("updated_at")
    normalized["overview_path"] = payload.get("overview_path")
    normalized["src_points_override"] = _normalize_src_points_override(payload.get("src_points_override"))
    normalized["dst_rect_override"] = _normalize_dst_rect_override(payload.get("dst_rect_override"))

    rois = []
    for index, roi in enumerate(payload.get("rois", []), start=1):
        xyxy = [int(value) for value in roi.get("xyxy", [0, 0, 0, 0])]
        rois.append(
            {
                "id": roi.get("id") or f"roi_{index:02d}",
                "xyxy": xyxy,
                "note": str(roi.get("note") or ""),
                "crop_path": roi.get("crop_path"),
            }
        )
    normalized["rois"] = rois

    points = []
    for index, point in enumerate(payload.get("points", []), start=1):
        points.append(
            {
                "id": point.get("id") or f"point_{index:02d}",
                "kind": str(point.get("kind") or "mark"),
                "label": str(point.get("label") or ""),
                "x": int(point.get("x", 0)),
                "y": int(point.get("y", 0)),
                "roi_id": point.get("roi_id"),
                "note": str(point.get("note") or ""),
            }
        )
    normalized["points"] = points

    lines = []
    for index, line in enumerate(payload.get("lines", []), start=1):
        lines.append(
            {
                "id": line.get("id") or f"line_{index:02d}",
                "kind": str(line.get("kind") or "horizontal_ref"),
                "label": str(line.get("label") or ""),
                "x1": int(line.get("x1", 0)),
                "y1": int(line.get("y1", 0)),
                "x2": int(line.get("x2", 0)),
                "y2": int(line.get("y2", 0)),
                "roi_id": line.get("roi_id"),
                "note": str(line.get("note") or ""),
                "real_distance": _optional_float(line.get("real_distance")),
                "coordinate_space": str(line.get("coordinate_space") or ""),
            }
        )
    normalized["lines"] = lines
    return normalized


def load_rois(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        legacy_json = path.with_suffix(".json")
        if legacy_json.exists():
            return _normalize_loaded_payload(json.loads(legacy_json.read_text(encoding="utf-8")))
        return default_payload()

    if path.suffix.lower() == ".toml":
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_loaded_payload(payload)


def _toml_string(value: str | None) -> str:
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def _format_number(value: object) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def _to_toml(payload: dict) -> str:
    lines = [
        f'image_name = {_toml_string(payload.get("image_name"))}',
        f'updated_at = {_toml_string(payload.get("updated_at"))}',
        f'overview_path = {_toml_string(payload.get("overview_path"))}',
    ]

    src_points_override = _normalize_src_points_override(payload.get("src_points_override"))
    if src_points_override:
        points_text = ", ".join(f"[{_format_number(x)}, {_format_number(y)}]" for x, y in src_points_override)
        lines.append(f"src_points_override = [{points_text}]")

    dst_rect_override = _normalize_dst_rect_override(payload.get("dst_rect_override"))
    if dst_rect_override:
        rect_text = ", ".join(_format_number(value) for value in dst_rect_override)
        lines.append(f"dst_rect_override = [{rect_text}]")

    for roi in payload.get("rois", []):
        xyxy = ", ".join(str(int(value)) for value in roi.get("xyxy", [0, 0, 0, 0]))
        lines.extend(
            [
                "",
                "[[rois]]",
                f'id = {_toml_string(roi.get("id"))}',
                f"xyxy = [{xyxy}]",
                f'crop_path = {_toml_string(roi.get("crop_path"))}',
                f'note = {_toml_string(roi.get("note"))}',
            ]
        )

    for point in payload.get("points", []):
        lines.extend(
            [
                "",
                "[[points]]",
                f'id = {_toml_string(point.get("id"))}',
                f'kind = {_toml_string(point.get("kind"))}',
                f'label = {_toml_string(point.get("label"))}',
                f'x = {int(point.get("x", 0))}',
                f'y = {int(point.get("y", 0))}',
                f'roi_id = {_toml_string(point.get("roi_id"))}',
                f'note = {_toml_string(point.get("note"))}',
            ]
        )

    for line in payload.get("lines", []):
        entry = [
            "",
            "[[lines]]",
            f'id = {_toml_string(line.get("id"))}',
            f'kind = {_toml_string(line.get("kind"))}',
            f'label = {_toml_string(line.get("label"))}',
            f'x1 = {int(line.get("x1", 0))}',
            f'y1 = {int(line.get("y1", 0))}',
            f'x2 = {int(line.get("x2", 0))}',
            f'y2 = {int(line.get("y2", 0))}',
            f'roi_id = {_toml_string(line.get("roi_id"))}',
            f'note = {_toml_string(line.get("note"))}',
        ]
        real_distance = _optional_float(line.get("real_distance"))
        if real_distance is not None:
            entry.append(f"real_distance = {real_distance}")
        coordinate_space = str(line.get("coordinate_space") or "")
        if coordinate_space:
            entry.append(f'coordinate_space = "{coordinate_space}"')
        lines.extend(entry)

    return "\n".join(lines) + "\n"


def save_rois(
    path: str | Path,
    image_path: str | Path,
    rois: list[dict],
    points: list[dict] | None = None,
    lines: list[dict] | None = None,
    src_points_override: list | None = None,
    dst_rect_override: list | tuple | dict | None = None,
) -> dict:
    path = Path(path)
    image_path = Path(image_path)
    ensure_dir(path.parent)
    points = points or []
    lines = lines or []

    crop_dir = path.parent / image_path.stem
    ensure_dir(crop_dir)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    overview = image_bgr.copy()
    saved_rois: list[dict] = []
    for index, roi in enumerate(rois, start=1):
        x1, y1, x2, y2 = [int(value) for value in roi["xyxy"]]
        note = str(roi.get("note") or "")
        crop = image_bgr[y1:y2, x1:x2]
        crop_name = f"roi_{index:02d}.jpg"
        crop_path = crop_dir / crop_name
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)

        cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(
            overview,
            f"ROI {index}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

        saved_rois.append(
            {
                "id": f"roi_{index:02d}",
                "xyxy": [x1, y1, x2, y2],
                "note": note,
                "crop_path": f"{image_path.stem}/{crop_name}",
            }
        )

    overview_path = crop_dir / "roi_overview.jpg"
    saved_points: list[dict] = []
    color_map = {
        "tube_top": (255, 200, 0),
        "tube_base": (0, 220, 255),
        "mark": (255, 80, 190),
    }
    for index, point in enumerate(points, start=1):
        x = int(point.get("x", 0))
        y = int(point.get("y", 0))
        kind = str(point.get("kind") or "mark")
        label = str(point.get("label") or f"point_{index:02d}")
        roi_id = point.get("roi_id")
        note = str(point.get("note") or "")
        color = color_map.get(kind, (160, 255, 160))

        cv2.circle(overview, (x, y), 6, color, -1, cv2.LINE_AA)
        cv2.circle(overview, (x, y), 11, color, 2, cv2.LINE_AA)
        cv2.putText(
            overview,
            label,
            (x + 10, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        saved_points.append(
            {
                "id": f"point_{index:02d}",
                "kind": kind,
                "label": label,
                "x": x,
                "y": y,
                "roi_id": roi_id,
                "note": note,
            }
        )

    saved_lines: list[dict] = []
    line_color_map = {
        "horizontal_ref": (80, 180, 255),
        "vertical_ref": (110, 110, 255),
        "tube_axis": (255, 120, 160),
        "custom": (120, 255, 140),
    }
    for index, line in enumerate(lines, start=1):
        x1 = int(line.get("x1", 0))
        y1 = int(line.get("y1", 0))
        x2 = int(line.get("x2", 0))
        y2 = int(line.get("y2", 0))
        kind = str(line.get("kind") or "horizontal_ref")
        label = str(line.get("label") or f"line_{index:02d}")
        roi_id = line.get("roi_id")
        note = str(line.get("note") or "")
        real_distance = _optional_float(line.get("real_distance"))
        color = line_color_map.get(kind, (80, 180, 255))

        cv2.line(overview, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(overview, (x1, y1), 5, color, -1, cv2.LINE_AA)
        cv2.circle(overview, (x2, y2), 5, color, -1, cv2.LINE_AA)
        label_x = int((x1 + x2) / 2)
        label_y = int((y1 + y2) / 2)
        cv2.putText(
            overview,
            label,
            (label_x + 8, max(20, label_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        saved_lines.append(
            {
                "id": f"line_{index:02d}",
                "kind": kind,
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "roi_id": roi_id,
                "note": note,
                "real_distance": real_distance,
            }
        )

    cv2.imwrite(str(overview_path), overview)

    payload = {
        "image_name": image_path.name,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "overview_path": f"{image_path.stem}/roi_overview.jpg",
        "src_points_override": _normalize_src_points_override(src_points_override),
        "dst_rect_override": _normalize_dst_rect_override(dst_rect_override),
        "rois": saved_rois,
        "points": saved_points,
        "lines": saved_lines,
    }
    path.write_text(_to_toml(payload), encoding="utf-8")
    return payload
