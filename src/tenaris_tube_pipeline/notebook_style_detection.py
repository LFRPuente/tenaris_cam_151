from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.cam151_ref_detection.homography_preview import build_homography_preview
from src.cam151_ref_detection.tube_detection_preview import (
    TubeDetectionPreviewResult,
    _compute_reference_measurements,
    _compute_reference_source_lines,
    _estimate_dominant_period,
    _estimate_stack_band,
    _extend_periodic_seams,
    _fill_missing_seams,
    _find_best_periodic_peak_run,
    _find_local_peaks_1d,
    _is_cam152_image,
    _line_x_at_y,
    _median_period,
    _mirror_line_source_x,
    _mirror_roi_source_x,
    _needs_backend_mirror,
    _normalize_01,
    _pick_detection_roi,
    _project_xy,
    _smooth_1d,
    _smooth_x_positions,
    _warp_detection_roi,
)


def _row_max_dark_run(mask_bool: np.ndarray) -> np.ndarray:
    runs: list[int] = []
    for row in np.asarray(mask_bool, bool):
        padded = np.r_[False, row, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        lengths = changes[1::2] - changes[::2]
        runs.append(int(lengths.max()) if lengths.size else 0)
    return np.asarray(runs, np.float32)


def _continuous_ranges(indices: np.ndarray, min_len: int = 8) -> list[tuple[int, int]]:
    values = np.asarray(indices, np.int32).reshape(-1)
    if values.size == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = int(values[0])
    for raw_value in values[1:].tolist():
        value = int(raw_value)
        if value == prev + 1:
            prev = value
            continue
        if prev - start + 1 >= min_len:
            ranges.append((start, prev))
        start = prev = value
    if prev - start + 1 >= min_len:
        ranges.append((start, prev))
    return ranges


def _merge_close_ranges(ranges: list[tuple[int, int]], max_gap: int = 120) -> list[tuple[int, int]]:
    if not ranges:
        return ranges
    merged = [list(ranges[0])]
    for y0, y1 in ranges[1:]:
        if y0 - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], y1)
        else:
            merged.append([y0, y1])
    return [(int(y0), int(y1)) for y0, y1 in merged]


def _find_point_by_label(points: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    wanted = str(label or "").strip().lower()
    for point in points:
        if str(point.get("label") or "").strip().lower() == wanted:
            return point
    return None


def _warp_point(transform: np.ndarray, point: dict[str, Any] | None) -> tuple[float, float] | None:
    if point is None:
        return None
    xy = np.asarray([[[float(point.get("x", 0.0)), float(point.get("y", 0.0))]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(xy, transform.astype(np.float32))
    if warped.size != 2:
        return None
    x_value, y_value = warped.reshape(-1).tolist()
    if not np.isfinite(x_value) or not np.isfinite(y_value):
        return None
    return float(x_value), float(y_value)


def _notebook_reference_lines(points: list[dict[str, Any]], transform: np.ndarray, *, cam152_mode: bool) -> list[dict[str, Any]]:
    top_point = _warp_point(transform, _find_point_by_label(points, "mark_02"))
    bottom_point = _warp_point(transform, _find_point_by_label(points, "mark_03"))
    if top_point is None or bottom_point is None:
        return []
    return [
        {
            "label": "ref_01" if cam152_mode else "ref_02",
            "x": float(0.5 * (top_point[0] + bottom_point[0])),
            "top_label": "mark_02",
            "bottom_label": "mark_03",
            "top_point": [float(top_point[0]), float(top_point[1])],
            "bottom_point": [float(bottom_point[0]), float(bottom_point[1])],
        }
    ]


def _notebook_reference_source_lines(points: list[dict[str, Any]], *, cam152_mode: bool) -> list[dict[str, Any]]:
    top_point = _find_point_by_label(points, "mark_02")
    bottom_point = _find_point_by_label(points, "mark_03")
    if top_point is None or bottom_point is None:
        return []
    return [
        {
            "label": "ref_01" if cam152_mode else "ref_02",
            "top_label": "mark_02",
            "bottom_label": "mark_03",
            "top_point": [float(top_point.get("x", 0.0)), float(top_point.get("y", 0.0))],
            "bottom_point": [float(bottom_point.get("x", 0.0)), float(bottom_point.get("y", 0.0))],
        }
    ]


def _pick_tube_stack_range(gray_img: np.ndarray) -> tuple[int, int, int, int, int, np.ndarray]:
    dark_thr = int(round(np.percentile(gray_img, 38)))
    dark_thr = max(95, min(170, dark_thr))
    dark_mask = gray_img <= dark_thr

    run_profile = _row_max_dark_run(dark_mask)
    run_profile_smooth = _smooth_1d(run_profile, 11)

    min_stack_y = int(round(0.10 * float(gray_img.shape[0])))
    min_dark_run = max(72, int(round(0.14 * float(gray_img.shape[1]))))

    active = run_profile_smooth >= float(min_dark_run)
    active[:min_stack_y] = False
    ranges = _continuous_ranges(np.flatnonzero(active), min_len=8)
    if not ranges:
        return 0, gray_img.shape[0] - 1, 0, gray_img.shape[1], dark_thr, run_profile_smooth

    ranges = _merge_close_ranges(ranges, max_gap=120)

    def range_score(rng: tuple[int, int]) -> float:
        y0, y1 = rng
        return float(y1 - y0 + 1) * float(np.median(run_profile_smooth[y0 : y1 + 1]))

    stack_y0, stack_y1 = max(ranges, key=range_score)
    margin = max(20, int(round(0.01 * float(gray_img.shape[0]))))
    stack_y0 = max(0, stack_y0 - margin)
    stack_y1 = min(gray_img.shape[0] - 1, stack_y1 + margin)

    ys, xs = np.where(dark_mask[stack_y0 : stack_y1 + 1])
    if xs.size:
        stack_x0 = max(0, int(np.percentile(xs, 3)) - 10)
        stack_x1 = min(gray_img.shape[1], int(np.percentile(xs, 97)) + 11)
    else:
        stack_x0, stack_x1 = 0, gray_img.shape[1]

    if stack_x1 <= stack_x0 + 24:
        stack_x0, stack_x1 = 0, gray_img.shape[1]

    return int(stack_y0), int(stack_y1), int(stack_x0), int(stack_x1), int(dark_thr), run_profile_smooth


def _detection_roi_from_notebook_flow(
    image_path: Path,
    roi_payload: dict[str, Any],
    homography_src_points: list[tuple[float, float]],
    output_size: tuple[int, int],
    warp_bgr: np.ndarray,
    *,
    final_transform: np.ndarray,
    mirror_in_backend: bool,
    source_width: int,
) -> tuple[int, int, int, int]:
    if roi_payload.get("dst_rect_override"):
        return (0, 0, int(warp_bgr.shape[1]), int(warp_bgr.shape[0]))

    selected_roi = _pick_detection_roi(list(roi_payload.get("rois") or []))
    selected_roi_processing = _mirror_roi_source_x(selected_roi, source_width) if mirror_in_backend else selected_roi
    detection_roi = _warp_detection_roi(
        selected_roi_processing,
        homography_src_points,
        output_size,
        warp_bgr.shape,
        output_flip_horizontal=mirror_in_backend,
    )
    if detection_roi is not None and not _is_cam152_image(image_path):
        stack_band = _estimate_stack_band(warp_bgr, detection_roi[0], detection_roi[2])
        if stack_band is not None:
            band_y0, band_y1, _run, period = stack_band
            top_expand = max(220, int(round(13.0 * float(period or 22.0))))
            detection_roi = (
                int(detection_roi[0]),
                int(min(detection_roi[1], max(0, band_y0 - top_expand))),
                int(detection_roi[2]),
                int(max(detection_roi[3], band_y1)),
            )
    if detection_roi is None:
        detection_roi = (0, 0, int(warp_bgr.shape[1]), int(warp_bgr.shape[0]))

    x0, y0, x1, y1 = [int(v) for v in detection_roi]
    x0 = max(0, min(x0, warp_bgr.shape[1] - 1))
    y0 = max(0, min(y0, warp_bgr.shape[0] - 1))
    x1 = max(x0 + 1, min(x1, warp_bgr.shape[1]))
    y1 = max(y0 + 1, min(y1, warp_bgr.shape[0]))
    return (x0, y0, x1, y1)


def _detect_notebook_positions(warp_crop: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(warp_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    tube_stack_y0, tube_stack_y1, strip_x0, strip_x1, _tube_dark_thr, _tube_run_profile = _pick_tube_stack_range(gray)
    gray_strip = gray[:, strip_x0:strip_x1]

    strip_blur_x = cv2.GaussianBlur(gray_strip, (9, 1), 0)
    dark_profile = 255.0 - np.percentile(strip_blur_x.astype(np.float32), 25, axis=1)
    grad_y = cv2.Sobel(cv2.GaussianBlur(gray_strip.astype(np.float32), (7, 3), 0), cv2.CV_32F, 0, 1, ksize=3)
    edge_profile = np.percentile(np.abs(grad_y), 80, axis=1)
    dark_score = _normalize_01(_smooth_1d(dark_profile, 7))
    edge_score = _normalize_01(_smooth_1d(edge_profile, 5))
    profile_norm = _normalize_01((0.65 * dark_score) + (0.35 * edge_score))
    profile_norm[:tube_stack_y0] = 0.0
    profile_norm[tube_stack_y1 + 1 :] = 0.0

    profile_cut = profile_norm[tube_stack_y0 : tube_stack_y1 + 1]
    dominant_period, _autocorr = _estimate_dominant_period(profile_cut, min_period_frac=0.04, max_period_frac=0.14)
    base_period = float(dominant_period) if dominant_period is not None and np.isfinite(dominant_period) else 22.0
    expected_tube_pitch = float(base_period)
    pitch_lo_preview = max(8.0, 0.60 * expected_tube_pitch)
    pitch_hi_preview = max(pitch_lo_preview + 1.0, 1.35 * expected_tube_pitch)
    base_min_distance = max(6, int(round(0.30 * base_period)))
    peaks_index = [
        int(tube_stack_y0 + peak)
        for peak in _find_local_peaks_1d(profile_cut, threshold=0.0, min_distance=base_min_distance)
    ]

    period_seed = float(dominant_period) if dominant_period is not None and np.isfinite(dominant_period) else None
    gap_lo_seed = int(round(pitch_lo_preview))
    gap_hi_seed = int(round(pitch_hi_preview))
    run_peaks, run_period = _find_best_periodic_peak_run(peaks_index, gap_lo=gap_lo_seed, gap_hi=gap_hi_seed, min_len=6)
    median_period = _median_period(run_peaks or peaks_index)
    period_used = run_period or median_period or period_seed or expected_tube_pitch or 22.0

    valid_observed_peaks = [int(p) for p in peaks_index if tube_stack_y0 <= int(p) <= tube_stack_y1]
    if len(valid_observed_peaks) >= 4:
        all_gaps = np.diff(np.asarray(sorted(valid_observed_peaks), np.float32))
        median_all_gap = float(np.median(all_gaps))
        if period_used > 1.6 * median_all_gap and median_all_gap >= 8.0:
            period_used = median_all_gap

    peaks_filled = _fill_missing_seams(valid_observed_peaks, period_used)
    peaks_filled = _extend_periodic_seams(
        peaks_filled,
        valid_observed_peaks,
        period_used,
        profile_norm,
        y_min=max(0, tube_stack_y0 - int(round(0.35 * float(period_used)))),
        y_max=min(int(profile_norm.size - 1), tube_stack_y1 + int(round(0.35 * float(period_used)))),
    )

    lo_gap = max(8.0, 0.60 * float(period_used))
    hi_gap = max(lo_gap + 1.0, 1.35 * float(period_used))

    def band_interval_ok(from_y: int, to_y: int) -> bool:
        gap = float(to_y - from_y)
        gap_in_range = lo_gap <= gap <= hi_gap
        both_inside_pack = (tube_stack_y0 <= float(from_y) <= tube_stack_y1) and (tube_stack_y0 <= float(to_y) <= tube_stack_y1)
        return bool(gap_in_range or both_inside_pack)

    tube_top_rows_list: list[int] = []
    tube_bottom_rows_list: list[int] = []
    rejected_tube_gaps: list[dict[str, Any]] = []
    for prev, curr in zip(peaks_filled[:-1], peaks_filled[1:]):
        if band_interval_ok(int(prev), int(curr)):
            tube_top_rows_list.append(int(prev))
            tube_bottom_rows_list.append(int(curr))
        else:
            gap = float(curr - prev)
            rejected_tube_gaps.append(
                {
                    "from_y": int(prev),
                    "to_y": int(curr),
                    "distance_px": float(gap),
                    "reason": "alto" if gap > hi_gap else "bajo",
                }
            )

    tube_top_rows = np.asarray(tube_top_rows_list, np.int32)
    tube_bottom_rows = np.asarray(tube_bottom_rows_list, np.int32)

    blur_kernel = (21, 1)
    x_search0 = max(int(round(0.32 * float(w))), strip_x0)
    x_search1 = max(x_search0 + 8, min(w, int(round(0.98 * float(w)))))
    roi_keep_frac = 5.0 / 8.0

    raw_x_positions: list[float] = []
    raw_hit_rows: list[int] = []
    roi_boxes: list[tuple[int, int]] = []
    for y_top, y_bot in zip(tube_top_rows, tube_bottom_rows):
        y0_full = max(0, int(y_top))
        y1 = min(gray.shape[0] - 1, int(y_bot))
        band_h = max(1, int(y1 - y0_full + 1))
        roi_h = max(3, min(band_h, int(round(roi_keep_frac * float(band_h)))))
        y0 = max(y0_full, int(y1 - roi_h + 1))
        fallback_x = float(raw_x_positions[-1]) if raw_x_positions else float(x_search0)
        hit_y = int(round(0.5 * float(y0 + y1)))
        x_pick = float(fallback_x)
        roi_boxes.append((int(y0), int(y1)))
        roi_gray = gray[y0 : y1 + 1, x_search0:x_search1]
        if roi_gray.size == 0:
            raw_x_positions.append(float(x_pick))
            raw_hit_rows.append(int(hit_y))
            continue

        roi_blur = cv2.GaussianBlur(roi_gray, blur_kernel, 0)
        roi_grad_x = cv2.Sobel(roi_blur.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        roi_grad_x_pos = np.maximum(roi_grad_x, 0.0)
        roi_grad_x_neg = np.maximum(-roi_grad_x, 0.0)
        roi_sobel_abs = np.abs(roi_grad_x).astype(np.float32)
        roi_sobel_abs = roi_sobel_abs - float(np.min(roi_sobel_abs))
        sobel_abs_den = float(np.max(roi_sobel_abs))
        if sobel_abs_den > 1e-6:
            roi_sobel_abs = roi_sobel_abs / sobel_abs_den
        else:
            roi_sobel_abs = np.zeros_like(roi_grad_x, np.float32)

        profile_raw = np.max(roi_sobel_abs, axis=0).astype(np.float32)
        profile_1d = np.power(_normalize_01(_smooth_1d(profile_raw, 9)), 12.0) if profile_raw.size else np.empty((0,), np.float32)
        x_local = None
        if profile_1d.size and float(np.max(profile_1d)) > 1e-6:
            peak_threshold = max(0.18, 0.45 * float(np.max(profile_1d)))
            cand_x = _find_local_peaks_1d(profile_1d, threshold=peak_threshold, min_distance=5)
            x_local = int(min(cand_x)) if cand_x else int(np.argmax(profile_1d))

        if x_local is not None:
            neg_strength = float(np.max(roi_grad_x_neg[:, x_local])) if roi_grad_x_neg.shape[0] > 0 else 0.0
            pos_strength = float(np.max(roi_grad_x_pos[:, x_local])) if roi_grad_x_pos.shape[0] > 0 else 0.0
            edge_col = roi_grad_x_neg[:, x_local].astype(np.float32) if neg_strength >= max(1e-6, 0.75 * pos_strength) else roi_grad_x_pos[:, x_local].astype(np.float32)
            edge_col = edge_col - float(np.min(edge_col))
            edge_den = float(np.max(edge_col))
            if edge_den > 1e-6:
                edge_col = edge_col / edge_den
            else:
                edge_col = np.zeros_like(edge_col, np.float32)

            cand_rows = np.flatnonzero(edge_col >= max(0.30, 0.70 * float(np.max(edge_col))))
            ref_rel_y = None
            if raw_hit_rows:
                ref_rel_y = float(np.median(np.asarray(raw_hit_rows[-3:], np.float32))) - float(y0)
            if cand_rows.size == 0:
                hit_local_y = int(np.argmax(edge_col)) if edge_col.size else int(round(0.5 * float(y1 - y0)))
            else:
                best_y_score = -1e9
                hit_local_y = int(cand_rows[0])
                for cand_y in cand_rows.tolist():
                    top_bonus = 1.0 - (float(cand_y) / max(1.0, float(edge_col.size - 1)))
                    consistency_y = 0.0
                    if ref_rel_y is not None:
                        consistency_y = max(0.0, 1.0 - abs(float(cand_y) - ref_rel_y) / max(4.0, 0.35 * float(edge_col.size)))
                    y_score = (1.00 * top_bonus) + (0.55 * consistency_y) + (0.25 * float(edge_col[cand_y]))
                    if y_score > best_y_score:
                        best_y_score = y_score
                        hit_local_y = int(cand_y)
            hit_y = int(y0 + hit_local_y)
            x_pick = float(x_search0 + x_local)

        raw_x_positions.append(float(x_pick))
        raw_hit_rows.append(int(hit_y))

    raw_pick_x_positions = list(raw_x_positions)
    smoothed_x_positions = _smooth_x_positions(raw_x_positions)
    probe_rows = np.asarray(raw_hit_rows, np.int32)

    x_start_list: list[dict[str, Any]] = []
    for pos, (x_raw, x_smooth, y_probe) in enumerate(zip(raw_pick_x_positions, smoothed_x_positions, probe_rows)):
        tube_idx = int(pos + 1)
        x_start_list.append(
            {
                "tube_idx": tube_idx,
                "x_start": float(x_raw),
                "x_local": float(x_raw),
                "x_end_estimate": float(x_raw),
                "x_seed": float(x_raw),
                "x_start_smooth": float(x_smooth),
                "y_center": float(y_probe),
                "confidence": 1.0,
            }
        )
    x_start_list.sort(key=lambda item: int(item["tube_idx"]))

    return {
        "gray": gray,
        "x_search0": x_search0,
        "x_search1": x_search1,
        "tube_stack_y0": tube_stack_y0,
        "tube_stack_y1": tube_stack_y1,
        "dominant_period": float(period_used) if period_used is not None else None,
        "energy_start_index": int(tube_stack_y0),
        "peaks_index": [int(v) for v in peaks_index],
        "peaks_index_dom": [int(v) for v in peaks_filled],
        "x_start_list": x_start_list,
        "roi_boxes": roi_boxes,
        "probe_rows": [int(v) for v in probe_rows.tolist()],
        "raw_pick_x_positions": [float(v) for v in raw_pick_x_positions],
        "smoothed_x_positions": [float(v) for v in smoothed_x_positions],
        "pitch_lo": float(lo_gap),
        "pitch_hi": float(hi_gap),
        "rejected_tube_gaps": rejected_tube_gaps,
    }


def _draw_detection_overlay(warp_bgr: np.ndarray, detection_roi: tuple[int, int, int, int], summary: dict[str, Any]) -> np.ndarray:
    overlay = warp_bgr.copy()
    x0, y0, x1, y1 = detection_roi
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2, cv2.LINE_AA)
    for y in summary["peaks_index_dom"]:
        cv2.line(overlay, (0, int(y) + y0), (overlay.shape[1] - 1, int(y) + y0), (0, 255, 120), 1, cv2.LINE_AA)
    for item in summary["x_start_list"]:
        x = int(round(float(item["x_start"]) + x0))
        y = int(round(float(item["y_center"]) + y0))
        cv2.circle(overlay, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(overlay, str(int(item["tube_idx"])), (x + 5, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def detect_tubes_like_notebook(
    image_path: str | Path,
    roi_payload: dict[str, Any],
    output_dir: str | Path,
) -> TubeDetectionPreviewResult:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = list(roi_payload.get("points") or [])
    lines = list(roi_payload.get("lines") or [])
    mirror_in_backend = _needs_backend_mirror(image_path)
    cam152_mode = _is_cam152_image(image_path)

    homography = build_homography_preview(
        image_path=image_path,
        lines=lines,
        points=points,
        output_dir=output_dir,
        src_points_override=roi_payload.get("src_points_override"),
        dst_rect_override=roi_payload.get("dst_rect_override"),
        flip_horizontal=mirror_in_backend,
        output_flip_horizontal=mirror_in_backend,
    )
    final_transform = cv2.getPerspectiveTransform(
        np.float32(homography.src_points),
        np.float32(
            [
                [0, 0],
                [homography.output_size[0] - 1, 0],
                [0, homography.output_size[1] - 1],
                [homography.output_size[0] - 1, homography.output_size[1] - 1],
            ]
        ),
    )
    if mirror_in_backend:
        output_flip = np.float32(
            [
                [-1.0, 0.0, float(homography.output_size[0] - 1)],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        final_transform = output_flip @ final_transform
    final_inverse_transform = np.linalg.inv(final_transform)

    warp_bgr = cv2.imread(str(homography.warp_path), cv2.IMREAD_COLOR)
    if warp_bgr is None or warp_bgr.size == 0:
        raise FileNotFoundError(f"Cannot read warp preview: {homography.warp_path}")

    detection_roi = _detection_roi_from_notebook_flow(
        image_path,
        roi_payload,
        homography.src_points,
        homography.output_size,
        warp_bgr,
        final_transform=final_transform,
        mirror_in_backend=mirror_in_backend,
        source_width=homography.source_size[0],
    )
    x0, y0, x1, y1 = detection_roi
    warp_crop = warp_bgr[y0:y1, x0:x1].copy()
    summary = _detect_notebook_positions(warp_crop)

    adjusted_x_starts: list[dict[str, Any]] = []
    for item in summary["x_start_list"]:
        adjusted = dict(item)
        for key in ("x_start", "x_local", "x_end_estimate", "x_seed", "x_start_smooth"):
            if adjusted.get(key) is not None:
                adjusted[key] = float(adjusted[key]) + x0
        adjusted["y_center"] = float(adjusted["y_center"]) + y0
        adjusted_x_starts.append(adjusted)
    summary["x_start_list"] = adjusted_x_starts

    if mirror_in_backend and homography.source_size[0] > 0:
        src_w = homography.source_size[0]
        points_for_references = [dict(point, x=int(src_w - 1 - int(point.get("x", 0)))) for point in points]
        lines_for_references = [_mirror_line_source_x(line, src_w) for line in lines]
    else:
        points_for_references = points
        lines_for_references = lines

    px_per_in, _computed_reference_lines, scale_samples = _compute_reference_measurements(
        points_for_references,
        final_transform.tolist(),
        lines=lines_for_references,
        cam152_mode=cam152_mode,
    )
    reference_lines = _notebook_reference_lines(points_for_references, final_transform, cam152_mode=cam152_mode)
    if not reference_lines:
        reference_lines = _computed_reference_lines
    reference_source_lines = _notebook_reference_source_lines(points_for_references, cam152_mode=cam152_mode)
    if not reference_source_lines:
        reference_source_lines = _compute_reference_source_lines(points_for_references, cam152_mode=cam152_mode)

    if reference_lines:
        for item in summary["x_start_list"]:
            x_val = float(item.get("x_start", item.get("x_local", 0.0)))
            y_val = float(item.get("y_center", 0.0))
            ref_distances: dict[str, dict[str, float]] = {}
            for ref in reference_lines:
                ref_x = _line_x_at_y(ref["top_point"], ref["bottom_point"], y_val)
                dx_px = float(x_val - ref_x)
                entry = {
                    "ref_x": ref_x,
                    "offset_px": dx_px,
                    "distance_px": abs(dx_px),
                }
                if px_per_in is not None and px_per_in > 1e-6:
                    entry["distance_in"] = abs(dx_px) / float(px_per_in)
                ref_distances[str(ref["label"])] = entry
            item["ref_distances"] = ref_distances

    overlay_bgr = _draw_detection_overlay(warp_bgr, detection_roi, summary)
    detection_overlay_path = output_dir / "tube_detection_warp.jpg"
    cv2.imwrite(str(detection_overlay_path), overlay_bgr)

    source_overlay_bgr = cv2.imread(str(homography.overlay_path), cv2.IMREAD_COLOR)
    source_overlay_path = output_dir / "tube_detection_overlay.jpg"
    source_overlay_dirty = False
    if source_overlay_bgr is not None and reference_source_lines:
        for idx, ref in enumerate(reference_source_lines, start=1):
            color = (255, 210, 80) if idx == 1 else (80, 220, 255)
            top_pt = ref["top_point"]
            bottom_pt = ref["bottom_point"]
            cv2.line(source_overlay_bgr, (int(round(top_pt[0])), int(round(top_pt[1]))), (int(round(bottom_pt[0])), int(round(bottom_pt[1]))), color, 3, cv2.LINE_AA)
        source_overlay_dirty = True
    if source_overlay_bgr is not None and reference_lines:
        ref = reference_lines[-1]
        for item in summary["x_start_list"]:
            x_val = float(item.get("x_start", item.get("x_local", 0.0)))
            y_val = float(item.get("y_center", 0.0))
            ref_x = _line_x_at_y(ref["top_point"], ref["bottom_point"], y_val)
            tube_src = _project_xy(final_inverse_transform, x_val, y_val)
            ref_src = _project_xy(final_inverse_transform, ref_x, y_val)
            if tube_src is not None and ref_src is not None:
                cv2.line(source_overlay_bgr, (int(round(tube_src[0])), int(round(tube_src[1]))), (int(round(ref_src[0])), int(round(ref_src[1]))), (0, 60, 255), 2, cv2.LINE_AA)
                source_overlay_dirty = True
    if source_overlay_bgr is not None and source_overlay_dirty:
        cv2.imwrite(str(source_overlay_path), source_overlay_bgr)
        homography.overlay_path = source_overlay_path

    return TubeDetectionPreviewResult(
        homography=homography,
        detection_overlay_path=detection_overlay_path,
        tube_count=int(len(summary["x_start_list"])),
        dominant_period=summary["dominant_period"],
        energy_start_index=int(summary["energy_start_index"]) + int(detection_roi[1]),
        peaks_index=[int(v) + int(detection_roi[1]) for v in summary["peaks_index"]],
        peaks_index_dom=[int(v) + int(detection_roi[1]) for v in summary["peaks_index_dom"]],
        x_start_list=list(summary["x_start_list"]),
        detection_roi=detection_roi,
        px_per_in=px_per_in,
        reference_lines=reference_lines,
        scale_samples=scale_samples,
        processing_mode="cam152" if cam152_mode else "cam151",
        processing_stage="notebook_style_sobel_x_roi",
        pitch_lo=float(summary["pitch_lo"]) if summary.get("pitch_lo") is not None else None,
        pitch_hi=float(summary["pitch_hi"]) if summary.get("pitch_hi") is not None else None,
        rejected_tube_gaps=list(summary.get("rejected_tube_gaps") or []),
    )
