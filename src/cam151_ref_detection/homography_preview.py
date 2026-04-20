from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class HomographyPreviewResult:
    overlay_path: Path
    warp_path: Path
    src_points: list[tuple[float, float]]
    output_size: tuple[int, int]
    base_size: tuple[int, int]
    used_line_labels: list[str]
    warp_padding: dict[str, int]
    dst_rect: tuple[float, float, float, float]
    homography_matrix: list[list[float]]
    inverse_homography_matrix: list[list[float]]


def _label_key(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_point(point: dict) -> dict:
    return {
        "kind": str(point.get("kind") or "mark"),
        "label": str(point.get("label") or ""),
        "x": float(point.get("x", 0)),
        "y": float(point.get("y", 0)),
    }


def _normalize_line(line: dict) -> dict:
    return {
        "kind": str(line.get("kind") or "horizontal_ref"),
        "label": str(line.get("label") or ""),
        "x1": float(line.get("x1", 0)),
        "y1": float(line.get("y1", 0)),
        "x2": float(line.get("x2", 0)),
        "y2": float(line.get("y2", 0)),
    }


def _normalize_padding(warp_padding: dict | None) -> dict[str, int]:
    payload = dict(warp_padding or {})
    normalized: dict[str, int] = {}
    for key in ("left", "right", "top", "bottom"):
        try:
            value = int(round(float(payload.get(key, 0))))
        except Exception:
            value = 0
        normalized[key] = max(0, value)
    return normalized


def _apply_default_padding(padding: dict[str, int], base_width: int, base_height: int) -> dict[str, int]:
    resolved = dict(padding)
    if resolved.get("left", 0) <= 0:
        resolved["left"] = max(24, int(round(0.06 * float(base_width))))
    if resolved.get("right", 0) <= 0:
        resolved["right"] = max(140, int(round(0.55 * float(base_width))))
    if resolved.get("top", 0) <= 0:
        resolved["top"] = max(0, int(round(0.01 * float(base_height))))
    if resolved.get("bottom", 0) <= 0:
        resolved["bottom"] = max(0, int(round(0.01 * float(base_height))))
    return resolved


def _normalize_src_points_override(src_points_override: list | None) -> list[tuple[float, float]] | None:
    if not src_points_override:
        return None
    if len(src_points_override) != 4:
        raise ValueError("src_points_override debe tener exactamente 4 puntos.")

    normalized: list[tuple[float, float]] = []
    for point in src_points_override:
        if isinstance(point, dict):
            x = float(point.get("x"))
            y = float(point.get("y"))
        else:
            x = float(point[0])
            y = float(point[1])
        normalized.append((x, y))
    return normalized


def _normalize_dst_rect_override(dst_rect_override: list | tuple | dict | None) -> tuple[float, float, float, float] | None:
    if not dst_rect_override:
        return None

    if isinstance(dst_rect_override, dict):
        x0 = float(dst_rect_override.get("x0"))
        y0 = float(dst_rect_override.get("y0"))
        x1 = float(dst_rect_override.get("x1"))
        y1 = float(dst_rect_override.get("y1"))
    else:
        if len(dst_rect_override) != 4:
            raise ValueError("dst_rect_override debe tener exactamente 4 valores.")
        x0 = float(dst_rect_override[0])
        y0 = float(dst_rect_override[1])
        x1 = float(dst_rect_override[2])
        y1 = float(dst_rect_override[3])

    if x1 <= x0 or y1 <= y0:
        raise ValueError("dst_rect_override es invalido; requiere x1 > x0 y y1 > y0.")
    return (x0, y0, x1, y1)


def _line_avg_y(line: dict) -> float:
    return (float(line["y1"]) + float(line["y2"])) / 2.0


def _line_avg_x(line: dict) -> float:
    return (float(line["x1"]) + float(line["x2"])) / 2.0


def _sorted_endpoints(line: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    p1 = (float(line["x1"]), float(line["y1"]))
    p2 = (float(line["x2"]), float(line["y2"]))
    if p1[0] <= p2[0]:
        return p1, p2
    return p2, p1


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def _label_for(item: dict, fallback: str) -> str:
    label = str(item.get("label") or "").strip()
    return label or fallback


def _pick_horizontal_refs(lines: Iterable[dict]) -> tuple[dict, dict]:
    horizontal_refs = [_normalize_line(line) for line in lines if str(line.get("kind")) == "horizontal_ref"]
    if len(horizontal_refs) < 2:
        raise ValueError("Se requieren al menos 2 lineas horizontal_ref para construir la homografia.")

    horizontal_refs.sort(key=_line_avg_y)
    return horizontal_refs[0], horizontal_refs[-1]


def _pick_vertical_refs(lines: Iterable[dict]) -> list[dict]:
    vertical_refs = [_normalize_line(line) for line in lines if str(line.get("kind")) == "vertical_ref"]
    vertical_refs.sort(key=_line_avg_x)
    return vertical_refs


def _find_line_by_labels(lines: list[dict], *labels: str) -> dict | None:
    wanted = {_label_key(label) for label in labels if label}
    if not wanted:
        return None

    for line in lines:
        if _label_key(line.get("label")) in wanted:
            return line
    return None


def _find_point_by_label(points: list[dict], label: str) -> dict | None:
    wanted = _label_key(label)
    for point in points:
        if _label_key(point.get("label")) == wanted:
            return point
    return None


def _point_xy(point: dict) -> tuple[float, float]:
    return float(point["x"]), float(point["y"])


def _translate_line_through_point(
    line: tuple[tuple[float, float], tuple[float, float]],
    anchor: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    (x1, y1), (x2, y2) = line
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        raise ValueError("No se puede trasladar una linea degenerada.")
    ax, ay = anchor
    return (anchor, (ax + dx, ay + dy))


def _line_x_at_y(
    line: tuple[tuple[float, float], tuple[float, float]],
    y_value: float,
) -> float:
    (x0, y0), (x1, y1) = line
    if abs(y1 - y0) < 1e-8:
        return float(0.5 * (x0 + x1))
    t = (float(y_value) - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def _has_secondary_parallel_guide_refs(points: list[dict], lines: list[dict]) -> bool:
    needed_points = {"top_03", "top_04", "mark_05", "mark_06"}
    present_points = {_label_key(point.get("label")) for point in points}
    if not needed_points.issubset(present_points):
        return False

    line_labels = {_label_key(line.get("label")) for line in lines}
    return "top_ref" in line_labels and ("bottom_horrizontal" in line_labels or "bottom_horizontal" in line_labels)


def _has_primary_parallel_guide_refs(points: list[dict], lines: list[dict]) -> bool:
    needed_points = {"top_01", "top_02", "mark_01", "mark_02"}
    present_points = {_label_key(point.get("label")) for point in points}
    if not needed_points.issubset(present_points):
        return False

    line_labels = {_label_key(line.get("label")) for line in lines}
    return "top_ref" in line_labels and ("bottom_horrizontal" in line_labels or "bottom_horizontal" in line_labels)


def _select_vertical_side_guides(
    lines: list[dict],
    left_seed: tuple[tuple[float, float], tuple[float, float]],
    right_seed: tuple[tuple[float, float], tuple[float, float]],
    y_probe: float,
) -> tuple[
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
    list[str],
    list[tuple[str, tuple[float, float], tuple[float, float]]],
] | None:
    vertical_refs = _pick_vertical_refs(lines)
    if len(vertical_refs) < 2:
        return None

    left_seed_x = _line_x_at_y(left_seed, y_probe)
    right_seed_x = _line_x_at_y(right_seed, y_probe)
    candidates: list[tuple[float, float, dict]] = []
    for line in vertical_refs:
        line_pts = _sorted_endpoints(line)
        x_val = _line_x_at_y(line_pts, y_probe)
        candidates.append((x_val, abs(x_val - left_seed_x), line))

    left_idx = int(np.argmin(np.asarray([item[1] for item in candidates], np.float32)))
    left_line = candidates[left_idx][2]
    left_pts = _sorted_endpoints(left_line)
    left_x = candidates[left_idx][0]

    right_candidates: list[tuple[float, float, dict]] = []
    for idx, (_, _, line) in enumerate(candidates):
        if idx == left_idx:
            continue
        line_pts = _sorted_endpoints(line)
        x_val = _line_x_at_y(line_pts, y_probe)
        right_candidates.append((x_val, abs(x_val - right_seed_x), line))
    if not right_candidates:
        return None

    right_idx = int(np.argmin(np.asarray([item[1] for item in right_candidates], np.float32)))
    right_line = right_candidates[right_idx][2]
    right_pts = _sorted_endpoints(right_line)
    right_x = right_candidates[right_idx][0]

    if left_x > right_x:
        left_line, right_line = right_line, left_line
        left_pts, right_pts = right_pts, left_pts

    debug_lines = [
        (_label_for(left_line, "vertical_ref_left"), left_pts[0], left_pts[1]),
        (_label_for(right_line, "vertical_ref_right"), right_pts[0], right_pts[1]),
    ]
    labels = [_label_for(left_line, "vertical_ref_left"), _label_for(right_line, "vertical_ref_right")]
    return left_pts, right_pts, labels, debug_lines


def _line_coeffs(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float, float]:
    x1, y1 = p1
    x2, y2 = p2
    return (y1 - y2, x2 - x1, x1 * y2 - x2 * y1)


def _intersect_lines(
    line_a: tuple[tuple[float, float], tuple[float, float]],
    line_b: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    a1, b1, c1 = _line_coeffs(*line_a)
    a2, b2, c2 = _line_coeffs(*line_b)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        raise ValueError("No se pudo construir la homografia: hay lineas paralelas o degeneradas.")
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return (float(x), float(y))


def _draw_point(canvas: np.ndarray, point: tuple[float, float], label: str, color: tuple[int, int, int]) -> None:
    x = int(round(point[0]))
    y = int(round(point[1]))
    cv2.circle(canvas, (x, y), 7, color, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        label,
        (x + 10, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_segment(
    canvas: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    cv2.line(
        canvas,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        color,
        thickness,
        cv2.LINE_AA,
    )


def _validate_quad(src_points: list[tuple[float, float]]) -> None:
    top_left, top_right, bottom_left, bottom_right = src_points
    top_width = _distance(top_left, top_right)
    bottom_width = _distance(bottom_left, bottom_right)
    left_height = _distance(top_left, bottom_left)
    right_height = _distance(top_right, bottom_right)

    if min(top_width, bottom_width, left_height, right_height) < 40:
        raise ValueError("La homografia quedo degenerada; revisa puntos y lineas.")


def _project_dst_rect_to_src(
    inverse_transform: np.ndarray,
    dst_rect: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = dst_rect
    dst_quad = np.array(
        [[[x0, y0]], [[x1, y0]], [[x0, y1]], [[x1, y1]]],
        dtype=np.float32,
    )
    src_quad = cv2.perspectiveTransform(dst_quad, inverse_transform)
    return [(float(p[0][0]), float(p[0][1])) for p in src_quad]


def _build_from_parallel_guides(
    points: list[dict],
    lines: list[dict],
) -> tuple[list[tuple[float, float]], list[str], dict]:
    top_left_point = _find_point_by_label(points, "top_01")
    top_right_point = _find_point_by_label(points, "top_02")
    mark_left_point = _find_point_by_label(points, "mark_01")
    mark_right_point = _find_point_by_label(points, "mark_02")
    top_line = _find_line_by_labels(lines, "top_ref")
    bottom_line = _find_line_by_labels(lines, "bottom_horrizontal", "bottom_horizontal")

    if not all([top_left_point, top_right_point, mark_left_point, mark_right_point, top_line, bottom_line]):
        raise ValueError("Faltan referencias etiquetadas: top_01, top_02, mark_01, mark_02, top_ref y bottom_horrizontal.")

    top_ref = _sorted_endpoints(top_line)
    bottom_ref = _sorted_endpoints(bottom_line)
    left_side = (_point_xy(top_left_point), _point_xy(mark_left_point))
    base_right_side = (_point_xy(top_right_point), _point_xy(mark_right_point))

    top_left = _intersect_lines(left_side, top_ref)
    bottom_left = _intersect_lines(left_side, bottom_ref)

    base_top_right = _intersect_lines(base_right_side, top_ref)
    base_bottom_right = _intersect_lines(base_right_side, bottom_ref)
    bottom_right_anchor = bottom_ref[1]
    if bottom_right_anchor[0] > base_bottom_right[0] + 2:
        right_side = _translate_line_through_point(base_right_side, bottom_right_anchor)
        bottom_right = bottom_right_anchor
        top_right = _intersect_lines(right_side, top_ref)
        right_label = "top_02->mark_02 + extend_right"
    else:
        right_side = base_right_side
        bottom_right = base_bottom_right
        top_right = base_top_right
        right_label = "top_02->mark_02"

    src_points = [top_left, top_right, bottom_left, bottom_right]
    _validate_quad(src_points)

    debug = {
        "top_ref": top_ref,
        "bottom_ref": bottom_ref,
        "left_side": left_side,
        "right_side": right_side,
        "anchor_points": [
            ("top_01", _point_xy(top_left_point)),
            ("top_02", _point_xy(top_right_point)),
            ("mark_01", _point_xy(mark_left_point)),
            ("mark_02", _point_xy(mark_right_point)),
            ("BR_ext", bottom_right_anchor),
        ],
    }
    labels = [
        _label_for(top_line, "top_ref"),
        _label_for(bottom_line, "bottom_horrizontal"),
        "top_01->mark_01",
        right_label,
    ]
    return src_points, labels, debug


def _build_from_secondary_parallel_guides(
    points: list[dict],
    lines: list[dict],
) -> tuple[list[tuple[float, float]], list[str], dict]:
    top_left_point = _find_point_by_label(points, "top_03")
    top_right_point = _find_point_by_label(points, "top_04")
    mark_left_point = _find_point_by_label(points, "mark_06")
    mark_right_point = _find_point_by_label(points, "mark_05")
    top_line = _find_line_by_labels(lines, "top_ref")
    bottom_line = _find_line_by_labels(lines, "bottom_horrizontal", "bottom_horizontal")

    if not all([top_left_point, top_right_point, mark_left_point, mark_right_point, top_line, bottom_line]):
        raise ValueError("Faltan referencias etiquetadas: top_03, top_04, mark_06, mark_05, top_ref y bottom_horrizontal.")

    top_ref = _sorted_endpoints(top_line)
    bottom_ref = _sorted_endpoints(bottom_line)
    left_seed = (_point_xy(top_left_point), _point_xy(mark_left_point))
    right_seed = (_point_xy(top_right_point), _point_xy(mark_right_point))

    y_probe = 0.5 * (_line_avg_y(top_line) + _line_avg_y(bottom_line))
    side_labels = ["top_03->mark_06", "top_04->mark_05"]
    picked = _select_vertical_side_guides(lines, left_seed, right_seed, y_probe)
    if picked is not None:
        left_side, right_side, side_labels, _ = picked
    else:
        left_side = left_seed
        right_side = right_seed

    top_left = _intersect_lines(left_side, top_ref)
    bottom_left = _intersect_lines(left_side, bottom_ref)
    top_right = _intersect_lines(right_side, top_ref)
    bottom_right = _intersect_lines(right_side, bottom_ref)

    src_points = [top_left, top_right, bottom_left, bottom_right]
    _validate_quad(src_points)

    anchor_points = [
        ("top_03", _point_xy(top_left_point)),
        ("top_04", _point_xy(top_right_point)),
        ("mark_06", _point_xy(mark_left_point)),
        ("mark_05", _point_xy(mark_right_point)),
    ]

    debug = {
        "top_ref": top_ref,
        "bottom_ref": bottom_ref,
        "left_side": left_side,
        "right_side": right_side,
        "anchor_points": anchor_points,
    }
    labels = [
        _label_for(top_line, "top_ref"),
        _label_for(bottom_line, "bottom_horrizontal"),
        side_labels[0],
        side_labels[1],
    ]
    return src_points, labels, debug


def _build_from_horizontal_lines(lines: list[dict]) -> tuple[list[tuple[float, float]], list[str], dict]:
    top_line, bottom_line = _pick_horizontal_refs(lines)
    top_left, top_right = _sorted_endpoints(top_line)
    bottom_left, bottom_right = _sorted_endpoints(bottom_line)
    src_points = [top_left, top_right, bottom_left, bottom_right]
    _validate_quad(src_points)

    debug = {
        "top_ref": (top_left, top_right),
        "bottom_ref": (bottom_left, bottom_right),
        "left_side": (top_left, bottom_left),
        "right_side": (top_right, bottom_right),
        "anchor_points": [],
    }
    labels = [_label_for(top_line, "top_ref"), _label_for(bottom_line, "bottom_ref")]
    return src_points, labels, debug


def build_homography_preview(
    image_path: str | Path,
    lines: list[dict],
    output_dir: str | Path,
    points: list[dict] | None = None,
    warp_padding: dict | None = None,
    src_points_override: list | None = None,
    dst_rect_override: list | tuple | dict | None = None,
) -> HomographyPreviewResult:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    normalized_lines = [_normalize_line(line) for line in lines]
    normalized_points = [_normalize_point(point) for point in (points or [])]
    padding = _normalize_padding(warp_padding)
    override_points = _normalize_src_points_override(src_points_override)
    override_rect = _normalize_dst_rect_override(dst_rect_override)

    if override_points is not None:
        computed_src_points = override_points
        top_left, top_right, bottom_left, bottom_right = computed_src_points
        used_labels = ["TL/TR", "BL/BR", "TL->BL", "TR->BR"]
        debug = {
            "top_ref": (top_left, top_right),
            "bottom_ref": (bottom_left, bottom_right),
            "left_side": (top_left, bottom_left),
            "right_side": (top_right, bottom_right),
            "anchor_points": [
                ("TL", top_left),
                ("TR", top_right),
                ("BL", bottom_left),
                ("BR", bottom_right),
            ],
        }
    elif _has_secondary_parallel_guide_refs(normalized_points, normalized_lines):
        computed_src_points, used_labels, debug = _build_from_secondary_parallel_guides(normalized_points, normalized_lines)
    elif _has_primary_parallel_guide_refs(normalized_points, normalized_lines):
        computed_src_points, used_labels, debug = _build_from_parallel_guides(normalized_points, normalized_lines)
    else:
        computed_src_points, used_labels, debug = _build_from_horizontal_lines(normalized_lines)

    base_src_points = computed_src_points
    _validate_quad(base_src_points)

    top_left, top_right, bottom_left, bottom_right = base_src_points
    top_width = _distance(top_left, top_right)
    bottom_width = _distance(bottom_left, bottom_right)
    left_height = _distance(top_left, bottom_left)
    right_height = _distance(top_right, bottom_right)

    base_width = max(240, int(round(max(top_width, bottom_width))))
    base_height = max(320, int(round(max(left_height, right_height))))
    if override_points is None:
        padding = _apply_default_padding(padding, base_width, base_height)
    else:
        padding = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    base_dst_points = np.float32(
        [
            [0, 0],
            [base_width - 1, 0],
            [0, base_height - 1],
            [base_width - 1, base_height - 1],
        ]
    )
    base_transform = cv2.getPerspectiveTransform(np.float32(base_src_points), base_dst_points)
    base_inverse_transform = np.linalg.inv(base_transform)

    if override_rect is None:
        if override_points is None:
            dst_rect = (
                float(-padding["left"]),
                float(-padding["top"]),
                float(base_width - 1 + padding["right"]),
                float(base_height - 1 + padding["bottom"]),
            )
        else:
            dst_rect = (0.0, 0.0, float(base_width - 1), float(base_height - 1))
    else:
        dst_rect = override_rect

    src_points = _project_dst_rect_to_src(base_inverse_transform, dst_rect)
    _validate_quad(src_points)

    x0, y0, x1, y1 = dst_rect
    out_width = max(80, int(round(x1 - x0)))
    out_height = max(80, int(round(y1 - y0)))
    dst_points = np.float32(
        [
            [0, 0],
            [out_width - 1, 0],
            [0, out_height - 1],
            [out_width - 1, out_height - 1],
        ]
    )
    final_transform = cv2.getPerspectiveTransform(np.float32(src_points), dst_points)
    warp = cv2.warpPerspective(image_bgr, final_transform, (out_width, out_height))

    overlay = image_bgr.copy()
    quad = np.array(
        [
            [int(round(top_left[0])), int(round(top_left[1]))],
            [int(round(top_right[0])), int(round(top_right[1]))],
            [int(round(bottom_right[0])), int(round(bottom_right[1]))],
            [int(round(bottom_left[0])), int(round(bottom_left[1]))],
        ],
        dtype=np.int32,
    )
    cv2.polylines(overlay, [quad], isClosed=True, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    top_color = (110, 220, 255)
    bottom_color = (120, 255, 160)
    side_color = (130, 170, 255)
    anchor_color = (255, 180, 90)

    _draw_segment(overlay, *debug["top_ref"], top_color, 3)
    _draw_segment(overlay, *debug["bottom_ref"], bottom_color, 3)
    _draw_segment(overlay, *debug["left_side"], side_color, 2)
    _draw_segment(overlay, *debug["right_side"], side_color, 2)

    for label, point in debug["anchor_points"]:
        _draw_point(overlay, point, label, anchor_color)

    _draw_point(overlay, top_left, "TL", (255, 0, 255))
    _draw_point(overlay, top_right, "TR", (255, 0, 255))
    _draw_point(overlay, bottom_left, "BL", (255, 0, 255))
    _draw_point(overlay, bottom_right, "BR", (255, 0, 255))

    cv2.putText(
        overlay,
        f"top={used_labels[0]}",
        (int(round(top_left[0])) + 10, max(24, int(round(top_left[1])) - 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        top_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"bottom={used_labels[1]}",
        (int(round(bottom_left[0])) + 10, max(24, int(round(bottom_left[1])) - 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        bottom_color,
        2,
        cv2.LINE_AA,
    )

    overlay_path = output_dir / "homography_overlay.jpg"
    warp_path = output_dir / "homography_warp.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(warp_path), warp)

    return HomographyPreviewResult(
        overlay_path=overlay_path,
        warp_path=warp_path,
        src_points=src_points,
        output_size=(out_width, out_height),
        base_size=(base_width, base_height),
        used_line_labels=used_labels,
        warp_padding=padding,
        dst_rect=dst_rect,
        homography_matrix=base_transform.tolist(),
        inverse_homography_matrix=base_inverse_transform.tolist(),
    )
