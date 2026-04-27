from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .homography_preview import HomographyPreviewResult, build_homography_preview


@dataclass
class TubeDetectionPreviewResult:
    homography: HomographyPreviewResult
    detection_overlay_path: Path
    tube_count: int
    dominant_period: float | None
    energy_start_index: int
    peaks_index: list[int]
    peaks_index_dom: list[int]
    x_start_list: list[dict]
    detection_roi: tuple[int, int, int, int] | None
    px_per_in: float | None
    reference_lines: list[dict]
    scale_samples: list[dict]
    processing_mode: str
    processing_stage: str
    pitch_lo: float | None
    pitch_hi: float | None
    rejected_tube_gaps: list[dict]


@lru_cache(maxsize=1)
def _get_sam2_predictor(model_id: str = "facebook/sam2.1-hiera-tiny"):
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SAM2ImagePredictor.from_pretrained(model_id, device=device)
    except Exception:
        return None


def _normalize_01(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, np.float32).reshape(-1)
    if x.size == 0:
        return x
    x = x - float(np.min(x))
    den = float(np.max(x)) + 1e-6
    return x / den


def _smooth_1d(signal: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(signal, np.float32).reshape(-1)
    if x.size == 0:
        return x
    k = max(3, int(window))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, np.float32) / float(k)
    return np.convolve(x, kernel, mode="same")


def _find_local_peaks_1d(signal: np.ndarray, threshold: float, min_distance: int) -> list[int]:
    x = np.asarray(signal, np.float32).reshape(-1)
    if x.size < 3:
        return []

    candidates: list[int] = []
    for idx in range(1, x.size - 1):
        if x[idx] >= threshold and x[idx] >= x[idx - 1] and x[idx] > x[idx + 1]:
            candidates.append(idx)

    candidates.sort(key=lambda idx: float(x[idx]), reverse=True)
    selected: list[int] = []
    min_distance = max(1, int(min_distance))
    for idx in candidates:
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(idx)
    selected.sort()
    return selected


def _label_key(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _find_point_by_label(points: list[dict], label: str) -> dict | None:
    wanted = _label_key(label)
    for point in points:
        if _label_key(point.get("label")) == wanted:
            return point
    return None


def _warp_point(transform: np.ndarray, point: dict) -> tuple[float, float] | None:
    if point is None:
        return None
    xy = np.array([[[float(point.get("x", 0.0)), float(point.get("y", 0.0))]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(xy, transform)
    if warped.size != 2:
        return None
    x, y = warped.reshape(-1).tolist()
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return (float(x), float(y))


def _project_xy(transform: np.ndarray, x_value: float, y_value: float) -> tuple[float, float] | None:
    xy = np.array([[[float(x_value), float(y_value)]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(xy, transform)
    if warped.size != 2:
        return None
    x_out, y_out = warped.reshape(-1).tolist()
    if not np.isfinite(x_out) or not np.isfinite(y_out):
        return None
    return (float(x_out), float(y_out))


def _line_is_warp_space(line: dict) -> bool:
    return str(line.get("kind") or "") == "warp_scale_ref" or str(line.get("coordinate_space") or "") == "warp"


def _compute_reference_measurements(
    points: list[dict],
    homography_matrix: list[list[float]],
    *,
    lines: list[dict] | None = None,
    cam152_mode: bool = False,
) -> tuple[float | None, list[dict], list[dict]]:
    transform = np.asarray(homography_matrix, dtype=np.float32)
    if transform.shape != (3, 3):
        return None, [], []

    scale_samples: list[dict] = []

    if True:
        # Escala desde lineas con real_distance anotado en el TOML.
        for line in (lines or []):
            dist_in = line.get("real_distance")
            if dist_in is None:
                continue
            is_warp_space = _line_is_warp_space(line)
            if is_warp_space:
                # Coordenadas ya en espacio warp — usar directamente sin transformar.
                x1w = float(line.get("x1", 0.0))
                y1w = float(line.get("y1", 0.0))
                x2w = float(line.get("x2", 0.0))
                y2w = float(line.get("y2", 0.0))
                dx_px = abs(x2w - x1w)
                p1 = [x1w, y1w]
                p2 = [x2w, y2w]
            else:
                p1_src = {"x": float(line.get("x1", 0.0)), "y": float(line.get("y1", 0.0))}
                p2_src = {"x": float(line.get("x2", 0.0)), "y": float(line.get("y2", 0.0))}
                p1 = _warp_point(transform, p1_src)
                p2 = _warp_point(transform, p2_src)
                if p1 is None or p2 is None:
                    continue
                dx_px = abs(float(p2[0] - p1[0]))
            if dx_px <= 1e-6:
                continue
            label = str(line.get("label") or line.get("id") or "line_ref")
            scale_samples.append(
                {
                    "label": label,
                    "distance_in": float(dist_in),
                    "distance_px": float(dx_px),
                    "px_per_in": float(dx_px / float(dist_in)),
                    "p1": [float(p1[0]), float(p1[1])],
                    "p2": [float(p2[0]), float(p2[1])],
                }
            )
    if not cam152_mode:
        # cam151: escala desde pares de puntos con distancia conocida.
        scale_specs = [
            ("top_03", "top_04", 38.75, "top_03-04"),
            ("mark_05", "mark_06", 38.75, "mark_05-06"),
            ("mark_03", "mark_04", 47.75, "mark_03-04"),
        ]
        for label_a, label_b, dist_in, name in scale_specs:
            point_a = _warp_point(transform, _find_point_by_label(points, label_a))
            point_b = _warp_point(transform, _find_point_by_label(points, label_b))
            if point_a is None or point_b is None:
                continue
            dx_px = abs(float(point_b[0] - point_a[0]))
            if dx_px <= 1e-6:
                continue
            scale_samples.append(
                {
                    "label": name,
                    "distance_in": float(dist_in),
                    "distance_px": float(dx_px),
                    "px_per_in": float(dx_px / float(dist_in)),
                    "p1": [float(point_a[0]), float(point_a[1])],
                    "p2": [float(point_b[0]), float(point_b[1])],
                }
            )

    px_per_in = None
    if scale_samples:
        px_per_in = float(np.median(np.asarray([sample["px_per_in"] for sample in scale_samples], np.float32)))

    # Lineas de referencia verticales: cam152 usa mark_02->mark_03; cam151 usa TR->BR.
    reference_lines: list[dict] = []
    ref_specs = (
        [("ref_01", "mark_02", "mark_03")]
        if cam152_mode
        else [("ref_02", "top_02", "base_02")]
    )
    for ref_label, top_label, bottom_label in ref_specs:
        top_pt = _warp_point(transform, _find_point_by_label(points, top_label))
        bottom_pt = _warp_point(transform, _find_point_by_label(points, bottom_label))
        if top_pt is not None and bottom_pt is not None:
            reference_lines.append(
                {
                    "label": ref_label,
                    "x": float(0.5 * (float(top_pt[0]) + float(bottom_pt[0]))),
                    "top_label": top_label,
                    "bottom_label": bottom_label,
                    "top_point": [float(top_pt[0]), float(top_pt[1])],
                    "bottom_point": [float(bottom_pt[0]), float(bottom_pt[1])],
                }
            )

    return px_per_in, reference_lines, scale_samples


def _compute_reference_source_lines(points: list[dict], *, cam152_mode: bool = False) -> list[dict]:
    ref_specs = (
        [("ref_01", "mark_02", "mark_03")]
        if cam152_mode
        else [("ref_02", "top_02", "base_02")]
    )
    source_lines: list[dict] = []
    for ref_label, top_label, bottom_label in ref_specs:
        top_pt = _find_point_by_label(points, top_label)
        bottom_pt = _find_point_by_label(points, bottom_label)
        if top_pt is None or bottom_pt is None:
            continue
        source_lines.append(
            {
                "label": ref_label,
                "top_label": top_label,
                "bottom_label": bottom_label,
                "top_point": [float(top_pt.get("x", 0.0)), float(top_pt.get("y", 0.0))],
                "bottom_point": [float(bottom_pt.get("x", 0.0)), float(bottom_pt.get("y", 0.0))],
            }
        )
    return source_lines


def _line_x_at_y(top_point: list[float], bottom_point: list[float], y_value: float) -> float:
    x0, y0 = float(top_point[0]), float(top_point[1])
    x1, y1 = float(bottom_point[0]), float(bottom_point[1])
    if abs(y1 - y0) <= 1e-6:
        return float(0.5 * (x0 + x1))
    t = (float(y_value) - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def _left_profile_strip_windows(width: int) -> list[tuple[int, int]]:
    w = max(1, int(width))
    frac_windows = [
        (0.00, 0.14),
        (0.03, 0.17),
        (0.06, 0.20),
    ]
    windows: list[tuple[int, int]] = []
    for fx0, fx1 in frac_windows:
        x0 = max(0, int(round(fx0 * float(w))))
        x1 = min(w, int(round(fx1 * float(w))))
        if x1 <= x0 + 20:
            x1 = min(w, x0 + 20)
        if x1 > x0 + 8:
            windows.append((x0, x1))

    deduped: list[tuple[int, int]] = []
    for win in windows:
        if win not in deduped:
            deduped.append(win)
    return deduped


def _build_left_multi_profile(gray_img: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int]]:
    gray = np.asarray(gray_img)
    if gray.size == 0:
        return np.asarray([], np.float32), [], (0, 0)

    h, w = gray.shape[:2]
    windows = _left_profile_strip_windows(w)
    profiles: list[np.ndarray] = []
    used: list[tuple[int, int]] = []
    for x0, x1 in windows:
        strip = gray[:, x0:x1]
        if strip.size == 0:
            continue
        blur_x = cv2.GaussianBlur(strip, (71, 1), 0)
        dark_profile = 255.0 - np.mean(blur_x.astype(np.float32), axis=1)
        profile = _normalize_01(_smooth_1d(dark_profile, 9))
        if profile.size != h:
            continue
        profiles.append(np.clip(profile.astype(np.float32), 1e-4, 1.0))
        used.append((int(x0), int(x1)))

    if not profiles:
        return np.asarray([], np.float32), [], (0, 0)

    stack = np.stack(profiles, axis=0)
    profile_med = np.median(stack, axis=0).astype(np.float32)
    profile_geo = np.exp(np.mean(np.log(stack), axis=0)).astype(np.float32)
    combined = _normalize_01((0.45 * profile_med) + (0.55 * profile_geo))
    x0 = min(v[0] for v in used)
    x1 = max(v[1] for v in used)
    return combined.astype(np.float32), used, (int(x0), int(x1))


def _estimate_dominant_period(signal: np.ndarray, min_period_frac: float = 0.02, max_period_frac: float = 0.50) -> tuple[float | None, np.ndarray | None]:
    x = np.asarray(signal, np.float32).reshape(-1)
    if x.size < 8:
        return None, None

    x = x - float(np.mean(x))
    autocorr = np.correlate(x, x, mode="full")[x.size - 1 :]
    autocorr = autocorr / (float(autocorr[0]) + 1e-9)

    min_period = max(4, int(round(float(min_period_frac) * float(x.size))))
    max_period = max(min_period + 1, int(round(float(max_period_frac) * float(x.size))))
    search = autocorr[min_period:max_period]
    if search.size == 0:
        return None, autocorr

    dominant_period = float(min_period + int(np.argmax(search)))
    return dominant_period, autocorr


def _select_bottom_peak_cluster(peaks: list[int], image_height: int, dominant_period: float | None) -> tuple[list[int], int | None]:
    ordered = sorted(int(p) for p in peaks)
    if len(ordered) < 8:
        return ordered, None

    diffs = np.diff(np.asarray(ordered, np.int32))
    if diffs.size == 0:
        return ordered, None

    period = float(dominant_period) if dominant_period is not None else float(np.median(diffs))
    lo = max(12, int(np.floor(0.75 * period)))
    hi = max(lo + 1, int(np.ceil(1.15 * period)))
    min_peaks = 8
    min_start_y = int(round(0.45 * float(image_height)))

    best: tuple[int, int, int] | None = None
    run_start = None
    for idx, diff in enumerate(diffs):
        ok = lo <= int(diff) <= hi
        if ok and run_start is None:
            run_start = idx
        elif not ok and run_start is not None:
            run_end = idx - 1
            peak_count = run_end - run_start + 2
            start_y = ordered[run_start]
            if peak_count >= min_peaks and start_y >= min_start_y:
                candidate = (peak_count, start_y, run_start)
                if best is None or candidate > best:
                    best = candidate
            run_start = None

    if run_start is not None:
        run_end = len(diffs) - 1
        peak_count = run_end - run_start + 2
        start_y = ordered[run_start]
        if peak_count >= min_peaks and start_y >= min_start_y:
            candidate = (peak_count, start_y, run_start)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return ordered, None

    _, _, start_idx = best
    return ordered[start_idx:], start_idx


def _median_period(peaks: list[int]) -> float | None:
    if len(peaks) < 3:
        return None
    diffs = np.diff(np.asarray(sorted(peaks), np.float32))
    valid = diffs[(diffs >= 12.0) & (diffs <= 40.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _fill_missing_seams(peaks: list[int], period: float | None) -> list[int]:
    ordered = sorted(int(p) for p in peaks)
    if len(ordered) < 2 or period is None or not np.isfinite(period) or period < 8.0:
        return ordered

    filled: list[int] = [ordered[0]]
    for current in ordered[1:]:
        prev = filled[-1]
        gap = float(current - prev)
        if gap > 1.65 * period:
            missing = int(round(gap / period)) - 1
            if missing > 0:
                step = gap / float(missing + 1)
                for miss_idx in range(missing):
                    filled.append(int(round(prev + step * float(miss_idx + 1))))
        filled.append(int(current))

    deduped: list[int] = []
    min_sep = max(10, int(round(0.55 * period)))
    for peak in filled:
        if not deduped or (peak - deduped[-1]) >= min_sep:
            deduped.append(peak)
    return deduped


def _smooth_x_positions(x_positions: list[float]) -> list[float]:
    if not x_positions:
        return []
    padded = [x_positions[0], *x_positions, x_positions[-1]]
    smoothed: list[float] = []
    for idx in range(1, len(padded) - 1):
        window = padded[idx - 1 : idx + 2]
        smoothed.append(float(np.median(np.asarray(window, np.float32))))
    return smoothed


def _median_smooth_positions(x_positions: list[float], radius: int = 2) -> list[float]:
    if not x_positions:
        return []
    rad = max(1, int(radius))
    out: list[float] = []
    n = len(x_positions)
    arr = np.asarray(x_positions, np.float32)
    for idx in range(n):
        lo = max(0, idx - rad)
        hi = min(n, idx + rad + 1)
        out.append(float(np.median(arr[lo:hi])))
    return out


def _normalize_profile(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, np.float32).reshape(-1)
    if x.size == 0:
        return x
    den = float(np.max(x)) + 1e-6
    return x / den


def _profile_window_median(profile: np.ndarray, x0: int, x1: int, default: float) -> float:
    arr = np.asarray(profile, np.float32).reshape(-1)
    lo = max(0, int(x0))
    hi = min(arr.size, int(x1))
    if hi <= lo:
        return float(default)
    return float(np.median(arr[lo:hi]))


def _find_threshold_crossing(
    profile_gray: np.ndarray,
    profile_neg: np.ndarray,
    lo: int,
    hi: int,
    body_level: float,
    dark_level: float,
) -> int | None:
    arr = np.asarray(profile_gray, np.float32).reshape(-1)
    neg = np.asarray(profile_neg, np.float32).reshape(-1)
    if arr.size == 0 or hi <= lo + 4:
        return None

    thr = float(body_level - 0.35 * max(6.0, body_level - dark_level))
    for idx in range(max(lo + 4, 0), min(hi - 6, arr.size - 6)):
        left_slice = arr[max(lo, idx - 6) : idx]
        right_slice = arr[idx : min(arr.size, idx + 5)]
        if left_slice.size < 4 or right_slice.size < 4:
            continue
        left_ok = float(np.mean(left_slice)) >= (thr + 3.0)
        right_ok = float(np.mean(right_slice)) <= thr
        neg_ok = float(np.max(neg[max(lo, idx - 1) : min(arr.size, idx + 2)])) >= 0.06
        if left_ok and right_ok and neg_ok:
            return int(idx)
    return None


def _pull_to_earliest_transition(
    candidate_x: float,
    threshold_cross: int | None,
    profile_gray: np.ndarray,
    body_level: float,
    dark_level: float,
) -> float:
    if threshold_cross is None:
        return float(candidate_x)

    cand_i = int(round(float(candidate_x)))
    cross_i = int(round(int(threshold_cross)))
    if cand_i <= cross_i + 6:
        return float(candidate_x)

    cross_body = _profile_window_median(profile_gray, cross_i - 56, cross_i - 26, default=body_level)
    cross_left = _profile_window_median(profile_gray, cross_i - 22, cross_i - 6, default=body_level)
    cross_right = _profile_window_median(profile_gray, cross_i + 4, cross_i + 18, default=dark_level)
    cross_drop = max(0.0, cross_left - cross_right)
    body_match = max(0.0, 1.0 - abs(cross_body - body_level) / max(16.0, 0.25 * body_level))
    if cross_drop < 10.0 or body_match < 0.22:
        return float(candidate_x)

    if cand_i - cross_i > 12:
        return float(cross_i)
    return float(0.85 * float(cross_i) + 0.15 * float(candidate_x))


def _sample_tube_body_lab(
    lab_img: np.ndarray | None,
    y_center: float,
    x_edge: float,
    *,
    band_half: int = 4,
    body_left: int = 92,
    body_right: int = 26,
) -> np.ndarray | None:
    if lab_img is None or lab_img.size == 0:
        return None

    h, w = lab_img.shape[:2]
    y0 = max(0, int(round(float(y_center))) - int(band_half))
    y1 = min(h - 1, int(round(float(y_center))) + int(band_half))
    x1 = max(0, int(round(float(x_edge))) - int(body_left))
    x2 = min(w, int(round(float(x_edge))) - int(body_right))
    if y1 <= y0 or x2 <= x1 + 10:
        return None

    patch = lab_img[y0 : y1 + 1, x1:x2]
    if patch.size == 0:
        return None
    return np.median(patch.reshape(-1, 3).astype(np.float32), axis=0)


def _local_search_bounds(
    x_seed: float,
    x_search0: int,
    x_search1: int,
    idx: int,
    total_count: int,
) -> tuple[int, int]:
    top_count = max(6, int(round(0.40 * float(max(1, total_count)))))
    if idx < top_count:
        frac = float(idx) / float(max(1, top_count - 1))
        left_margin = 36.0 + 8.0 * frac
        right_margin = 4.0 + 3.0 * frac
    else:
        left_margin = 46.0
        right_margin = 16.0

    lo = max(int(x_search0), int(round(float(x_seed) - left_margin)))
    hi = min(int(x_search1), int(round(float(x_seed) + right_margin)))
    if hi <= lo + 12:
        hi = min(int(x_search1), lo + 12)
    return lo, hi


def _lab_color_distance(sample_lab: np.ndarray | None, ref_lab: np.ndarray | None) -> float:
    if sample_lab is None or ref_lab is None:
        return 999.0
    sample = np.asarray(sample_lab, np.float32).reshape(3)
    ref = np.asarray(ref_lab, np.float32).reshape(3)
    diff = sample - ref
    weights = np.asarray([0.20, 1.0, 1.0], np.float32)
    return float(np.sqrt(np.sum((diff * weights) ** 2)))


def _estimate_tube_reference_lab(
    color_img: np.ndarray | None,
    centers: np.ndarray,
    x_positions: list[float],
) -> np.ndarray | None:
    if color_img is None or color_img.size == 0 or len(x_positions) == 0:
        return None
    if color_img.ndim == 2:
        color_bgr = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)
    else:
        color_bgr = color_img
    lab_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)

    x_arr = np.asarray(x_positions, np.float32)
    median_x = float(np.median(x_arr))
    n = int(x_arr.size)
    lo_idx = int(round(0.15 * float(n)))
    hi_idx = max(lo_idx + 1, int(round(0.85 * float(n))))

    samples: list[np.ndarray] = []
    fallback_samples: list[np.ndarray] = []
    for idx, (y_center, x_edge) in enumerate(zip(centers.tolist(), x_positions)):
        sample = _sample_tube_body_lab(lab_img, float(y_center), float(x_edge))
        if sample is not None:
            fallback_samples.append(sample)
            if lo_idx <= idx < hi_idx and abs(float(x_edge) - median_x) <= 28.0:
                samples.append(sample)

    use_samples = samples if len(samples) >= 4 else fallback_samples
    if len(use_samples) < 4:
        return None
    sample_arr = np.asarray(use_samples, np.float32)
    return np.median(sample_arr, axis=0)


def _refine_x_positions_with_local_body_mask(
    gray_img: np.ndarray,
    color_img: np.ndarray | None,
    centers: np.ndarray,
    base_x_positions: list[float],
    x_search0: int,
    x_search1: int,
) -> tuple[list[float], list[float]]:
    if (
        gray_img is None
        or gray_img.size == 0
        or color_img is None
        or color_img.size == 0
        or len(base_x_positions) == 0
    ):
        return base_x_positions, [0.0 for _ in base_x_positions]

    if color_img.ndim == 2:
        color_bgr = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)
    else:
        color_bgr = color_img
    lab_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    tube_ref_lab = _estimate_tube_reference_lab(color_bgr, centers, base_x_positions)

    refined: list[float] = []
    confidences: list[float] = []
    band_half = 6
    for y_center, x_seed in zip(centers.tolist(), base_x_positions):
        y0 = max(0, int(round(float(y_center))) - band_half)
        y1 = min(gray_img.shape[0] - 1, int(round(float(y_center))) + band_half)
        lo = max(int(x_search0), int(round(float(x_seed) - 116.0)))
        hi = min(int(x_search1), int(round(float(x_seed) + 42.0)))
        if y1 <= y0 or hi <= lo + 36:
            refined.append(float(x_seed))
            confidences.append(0.0)
            continue

        local_ref = _sample_tube_body_lab(
            lab_img,
            float(y_center),
            float(x_seed),
            band_half=band_half,
            body_left=108,
            body_right=46,
        )
        ref_lab = local_ref if local_ref is not None else tube_ref_lab
        if ref_lab is None:
            refined.append(float(x_seed))
            confidences.append(0.0)
            continue

        patch_lab = lab_img[y0 : y1 + 1, lo:hi].astype(np.float32)
        patch_gray = gray_img[y0 : y1 + 1, lo:hi].astype(np.float32)
        if patch_lab.size == 0 or patch_gray.size == 0:
            refined.append(float(x_seed))
            confidences.append(0.0)
            continue

        ref_vec = np.asarray(ref_lab, np.float32).reshape(1, 1, 3)
        diff = patch_lab - ref_vec
        dist = np.sqrt(
            (0.18 * diff[:, :, 0]) ** 2
            + diff[:, :, 1] ** 2
            + diff[:, :, 2] ** 2
        )

        ref_x1 = max(lo, int(round(float(x_seed) - 104.0))) - lo
        ref_x2 = min(hi, int(round(float(x_seed) - 52.0))) - lo
        if ref_x2 <= ref_x1 + 10:
            ref_x1 = max(0, int(round(0.06 * float(hi - lo))))
            ref_x2 = min(hi - lo, ref_x1 + 36)
        body_patch = patch_gray[:, ref_x1:ref_x2]
        body_gray = float(np.median(body_patch)) if body_patch.size else float(np.median(patch_gray[:, : max(8, int(round(0.20 * float(hi - lo))))]))
        bright_thr = body_gray - 14.0
        dist_thr = 10.5

        mask = (dist <= dist_thr) & (patch_gray >= bright_thr)
        support = _smooth_1d(mask.mean(axis=0).astype(np.float32), 7)
        if support.size < 24:
            refined.append(float(x_seed))
            confidences.append(0.0)
            continue

        best_x = None
        best_conf = 0.0
        for idx in range(support.size - 18, 10, -1):
            left_support = float(np.mean(support[max(0, idx - 12) : max(0, idx - 2)]))
            right_support = float(np.mean(support[idx + 3 : min(support.size, idx + 16)]))
            center_support = float(support[idx])
            drop = left_support - right_support
            candidate_x = float(lo + idx)
            if center_support >= 0.34 and left_support >= 0.44 and right_support <= 0.18 and drop >= 0.28:
                if candidate_x > float(x_seed) + 8.0:
                    continue
                conf_val = np.clip(
                    0.40 * min(1.0, center_support / 0.55)
                    + 0.35 * min(1.0, left_support / 0.70)
                    + 0.25 * min(1.0, drop / 0.55),
                    0.0,
                    1.0,
                )
                best_x = candidate_x
                best_conf = float(conf_val)
                break

        if best_x is None:
            deriv = _smooth_1d(np.diff(support, prepend=support[0]), 5)
            best_idx = None
            best_score = -1e9
            for idx in range(12, support.size - 16):
                left_support = float(np.mean(support[max(0, idx - 10) : idx]))
                right_support = float(np.mean(support[idx + 3 : min(support.size, idx + 14)]))
                candidate_x = float(lo + idx)
                if candidate_x > float(x_seed) + 8.0:
                    continue
                drop = left_support - right_support
                score = (
                    1.2 * drop
                    - 0.7 * float(deriv[idx])
                    + 0.15 * left_support
                    - 0.012 * max(0.0, float(x_seed) - candidate_x - 36.0)
                )
                if left_support >= 0.36 and drop >= 0.18 and score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None:
                best_x = float(lo + best_idx)
                best_conf = float(
                    np.clip(
                        0.35 * min(1.0, float(np.mean(support[max(0, best_idx - 10) : best_idx])) / 0.65)
                        + 0.35 * min(1.0, max(0.0, float(np.mean(support[max(0, best_idx - 10) : best_idx])) - float(np.mean(support[best_idx + 3 : min(support.size, best_idx + 14)]))) / 0.50)
                        + 0.30 * min(1.0, max(0.0, -float(deriv[best_idx])) / 0.10),
                        0.0,
                        1.0,
                    )
                )

        if best_x is None:
            refined.append(float(x_seed))
            confidences.append(0.0)
            continue

        refined.append(float(best_x))
        confidences.append(float(best_conf))

    return refined, confidences


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (np.asarray(mask, np.uint8) > 0).astype(np.uint8)
    if mask_u8.size == 0 or int(mask_u8.sum()) == 0:
        return mask_u8.astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    return labels == best_idx


def _score_sam2_mask(mask: np.ndarray, sam_score: float) -> float:
    mask_bool = np.asarray(mask, bool)
    if mask_bool.size == 0 or int(mask_bool.sum()) == 0:
        return -1e9

    h, w = mask_bool.shape[:2]
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return -1e9

    rows = mask_bool.mean(axis=1).astype(np.float32)
    active_rows = np.flatnonzero(rows > 0.10)
    top = int(active_rows[0]) if active_rows.size else int(ys.min())
    upper_support = float(rows[: max(1, int(round(0.45 * float(h))))].mean())
    lower_support = float(rows[int(round(0.55 * float(h))) :].mean())
    left_touch = float(np.mean(xs < int(round(0.10 * float(w)))))
    width_cov = float(np.percentile(xs, 95) - np.percentile(xs, 5)) / max(1.0, float(w))

    return (
        3.6 * lower_support
        + 2.1 * upper_support
        + 0.4 * float(sam_score)
        + 0.35 * left_touch
        + 0.25 * width_cov
        - 0.010 * float(top)
    )


def _segment_tube_block_sam2(crop_bgr: np.ndarray) -> dict | None:
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    if h < 80 or w < 80:
        return None

    predictor = _get_sam2_predictor()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    prompt_configs = [
        dict(box_top=0.38, top_pos_1=0.18, top_pos_2=0.20),
        dict(box_top=0.34, top_pos_1=0.18, top_pos_2=0.22),
    ]

    best: dict | None = None
    for cfg in prompt_configs:
        box = np.array(
            [
                max(8, int(round(0.04 * float(w)))),
                int(round(cfg["box_top"] * float(h))),
                max(24, int(round(0.98 * float(w)))),
                max(24, h - 6),
            ],
            dtype=np.float32,
        )
        pts = np.array(
            [
                [int(round(0.18 * float(w))), int(round(0.82 * float(h)))],
                [int(round(0.42 * float(w))), int(round(0.76 * float(h)))],
                [int(round(0.70 * float(w))), int(round(0.70 * float(h)))],
                [int(round(0.18 * float(w))), int(round(cfg["top_pos_1"] * float(h)))],
                [int(round(0.45 * float(w))), int(round(cfg["top_pos_2"] * float(h)))],
                [int(round(0.50 * float(w))), int(round(0.12 * float(h)))],
                [int(round(0.80 * float(w))), int(round(0.16 * float(h)))],
                [int(round(0.92 * float(w))), int(round(0.34 * float(h)))],
                [int(round(0.96 * float(w))), int(round(0.60 * float(h)))],
            ],
            dtype=np.float32,
        )
        lbs = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=lbs,
            box=box,
            multimask_output=True,
        )
        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)

        for mask, score in zip(masks, scores):
            mask_bool = _keep_largest_component(mask > 0)
            mask_u8 = (mask_bool.astype(np.uint8) * 255)
            mask_u8 = cv2.morphologyEx(
                mask_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5)),
            )
            mask_u8 = cv2.morphologyEx(
                mask_u8,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
            )
            mask_bool = _keep_largest_component(mask_u8 > 0)
            heuristic = _score_sam2_mask(mask_bool, float(score))
            if best is None or heuristic > float(best["heuristic"]):
                best = {
                    "mask": mask_bool,
                    "heuristic": float(heuristic),
                    "sam_score": float(score),
                    "box": box.copy(),
                    "points": pts.copy(),
                    "labels": lbs.copy(),
                }

    return best


def _refine_tube_edge_with_local_sam2(crop_bgr: np.ndarray) -> tuple[float, float] | None:
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    if h < 16 or w < 48:
        return None

    predictor = _get_sam2_predictor()
    if predictor is None:
        return None

    try:
        predictor.set_image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        return None

    box = np.array(
        [
            max(2, int(round(0.05 * float(w)))),
            max(2, int(round(0.10 * float(h)))),
            max(8, int(round(0.98 * float(w)))),
            max(8, int(round(0.90 * float(h)))),
        ],
        dtype=np.float32,
    )
    pts = np.array(
        [
            [int(round(0.22 * float(w))), int(round(0.50 * float(h)))],
            [int(round(0.42 * float(w))), int(round(0.50 * float(h)))],
            [int(round(0.86 * float(w))), int(round(0.50 * float(h)))],
            [int(round(0.56 * float(w))), int(round(0.14 * float(h)))],
            [int(round(0.56 * float(w))), int(round(0.86 * float(h)))],
            [int(round(0.94 * float(w))), int(round(0.22 * float(h)))],
            [int(round(0.94 * float(w))), int(round(0.78 * float(h)))],
        ],
        dtype=np.float32,
    )
    lbs = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.int32)

    try:
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=lbs,
            box=box,
            multimask_output=True,
        )
    except Exception:
        return None
    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)

    center_y = 0.50 * float(h)
    best_edge = None
    best_score = -1e9
    for mask, score in zip(masks, scores):
        mask_bool = _keep_largest_component(mask > 0)
        if mask_bool.size == 0 or int(mask_bool.sum()) == 0:
            continue
        y0 = max(0, int(round(center_y)) - 4)
        y1 = min(h - 1, int(round(center_y)) + 4)
        band = mask_bool[y0 : y1 + 1]
        rights: list[float] = []
        widths: list[float] = []
        for row in band:
            cols = np.flatnonzero(row)
            if cols.size >= 10:
                rights.append(float(np.percentile(cols.astype(np.float32), 92)))
                widths.append(float(np.percentile(cols.astype(np.float32), 92) - np.percentile(cols.astype(np.float32), 12)))
        if not rights:
            continue
        edge_x = float(np.median(np.asarray(rights, np.float32)))
        width_med = float(np.median(np.asarray(widths, np.float32))) if widths else 0.0
        score_local = float(score) + 0.004 * width_med - 0.006 * max(0.0, edge_x - 0.90 * float(w))
        if score_local > best_score:
            best_score = score_local
            best_edge = edge_x

    if best_edge is None:
        return None
    return float(best_edge), float(best_score)


def _refine_tube_edge_with_local_sobel(crop_bgr: np.ndarray) -> tuple[float, float] | None:
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 40:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (21, 1), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)

    grad_pos = np.maximum(grad_x, 0.0)
    grad_neg = np.maximum(-grad_x, 0.0)
    edge = np.maximum(grad_neg, 0.45 * grad_pos)
    edge = edge - float(np.min(edge))
    edge_den = float(np.max(edge))
    if edge_den <= 1e-6:
        return None
    edge = edge / edge_den

    row_peak = np.max(edge, axis=1, keepdims=True)
    support_mask = edge >= np.maximum(0.18, 0.55 * row_peak)
    support_count = np.sum(support_mask.astype(np.float32), axis=0)
    energy_sum = np.sum(edge, axis=0).astype(np.float32)
    profile = energy_sum * (0.30 + 0.70 * _normalize_01(_smooth_1d(support_count, 5)))
    profile = _normalize_01(_smooth_1d(profile, 7))
    if profile.size == 0:
        return None

    x_idx = int(np.argmax(profile))
    score = float(profile[x_idx])
    if score < 0.18:
        return None
    return float(x_idx), score


def _refine_tube_edge_with_local_felzenszwalb(crop_bgr: np.ndarray) -> tuple[float, float] | None:
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 40:
        return None

    try:
        from skimage.measure import regionprops
        from skimage.segmentation import felzenszwalb
    except Exception:
        return _refine_tube_edge_with_local_sobel(crop_bgr)

    smooth = crop_bgr.copy()
    for _ in range(2):
        smooth = cv2.edgePreservingFilter(smooth, flags=1, sigma_s=25, sigma_r=0.18)
    rgb = cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB)
    seg = felzenszwalb(rgb, scale=80, sigma=0.8, min_size=24)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    body_gray = float(np.median(gray[:, : max(10, int(round(0.28 * float(w))))]))
    best_edge = None
    best_score = -1e9
    for region in regionprops(seg + 1):
        minr, minc, maxr, maxc = region.bbox
        bh = maxr - minr
        bw = maxc - minc
        if bw < 18 or bh < 4:
            continue
        aspect = float(bw) / max(1.0, float(bh))
        if aspect < 2.2:
            continue
        mask = seg == (region.label - 1)
        if not np.any(mask):
            continue

        rows_active = np.any(mask, axis=1)
        row_cov = float(np.mean(rows_active))
        left_touch = float(np.mean(mask[:, : max(3, int(round(0.10 * float(w))))]))
        if row_cov < 0.55 or left_touch < 0.20:
            continue

        cols = np.flatnonzero(np.any(mask, axis=0))
        if cols.size < 8:
            continue
        right_x = float(np.percentile(cols.astype(np.float32), 96))
        comp_gray = float(np.median(gray[mask]))
        bright_score = max(0.0, 1.0 - abs(comp_gray - body_gray) / max(18.0, 0.18 * body_gray))
        right_penalty = max(0.0, right_x - 0.90 * float(w))
        score = (
            0.85 * row_cov
            + 0.55 * left_touch
            + 0.28 * min(1.0, aspect / 7.0)
            + 0.22 * bright_score
            - 0.010 * right_penalty
        )
        if score > best_score:
            best_score = score
            best_edge = right_x

    if best_edge is None:
        return None
    return float(best_edge), float(best_score)


def _mask_band_bounds(mask: np.ndarray | None, y_center: float, band_half: int = 4) -> tuple[float, float] | None:
    if mask is None or mask.size == 0:
        return None

    mask_bool = np.asarray(mask, bool)
    h, _w = mask_bool.shape[:2]
    y0 = max(0, int(round(float(y_center))) - int(band_half))
    y1 = min(h - 1, int(round(float(y_center))) + int(band_half))
    band = mask_bool[y0 : y1 + 1]
    if band.size == 0:
        return None

    row_lefts: list[float] = []
    row_rights: list[float] = []
    for row in band:
        cols = np.flatnonzero(row)
        if cols.size >= 10:
            cols_f = cols.astype(np.float32)
            row_lefts.append(float(np.percentile(cols_f, 8)))
            row_rights.append(float(np.percentile(cols_f, 92)))

    if not row_rights:
        return None

    return (
        float(np.median(np.asarray(row_lefts, np.float32))),
        float(np.median(np.asarray(row_rights, np.float32))),
    )


def _stabilize_x_positions(centers: np.ndarray, x_positions: list[float]) -> list[float]:
    if len(x_positions) < 5:
        return x_positions

    arr = np.asarray(x_positions, np.float32)
    guide_mid = np.asarray(_median_smooth_positions(x_positions, radius=2), np.float32)
    guide_wide = np.asarray(_median_smooth_positions(x_positions, radius=4), np.float32)
    guide = 0.45 * guide_mid + 0.55 * guide_wide

    corrected = arr.copy()
    n = int(arr.size)
    top_count = max(6, int(round(0.35 * float(n))))
    for idx in range(n):
        dev = float(arr[idx] - guide[idx])
        is_top = idx < top_count
        is_outlier = abs(dev) > 24.0 or (is_top and dev > 16.0)
        if is_outlier:
            corrected[idx] = float(0.25 * arr[idx] + 0.75 * guide[idx])

    try:
        ys = np.asarray(centers, np.float32).reshape(-1)
        sel = np.abs(corrected - guide) <= 18.0
        if int(np.count_nonzero(sel)) >= max(8, int(round(0.45 * float(n)))):
            coef = np.polyfit(ys[sel], corrected[sel], deg=1)
            trend = np.polyval(coef, ys).astype(np.float32)
            for idx in range(n):
                dev = float(corrected[idx] - trend[idx])
                is_top = idx < top_count
                if abs(dev) > 26.0 or (is_top and dev > 18.0):
                    corrected[idx] = float(0.20 * corrected[idx] + 0.80 * trend[idx])
    except Exception:
        pass

    return _smooth_x_positions(corrected.tolist())


def _fit_tip_trend(centers: np.ndarray, x_positions: list[float]) -> np.ndarray | None:
    if len(x_positions) < 6:
        return None

    ys = np.asarray(centers, np.float32).reshape(-1)
    xs = np.asarray(x_positions, np.float32).reshape(-1)
    if ys.size != xs.size or ys.size < 6:
        return None

    sel = np.ones(xs.size, dtype=bool)
    trend = None
    for _ in range(4):
        if int(np.count_nonzero(sel)) < 5:
            break
        coef = np.polyfit(ys[sel], xs[sel], deg=1)
        trend = np.polyval(coef, ys).astype(np.float32)
        resid = xs - trend
        mad = float(np.median(np.abs(resid[sel]))) + 1e-6
        thr = max(12.0, 2.8 * mad)
        new_sel = np.abs(resid) <= thr
        if int(np.count_nonzero(new_sel)) == int(np.count_nonzero(sel)):
            sel = new_sel
            break
        sel = new_sel

    if trend is None:
        return None

    if int(np.count_nonzero(sel)) >= 5:
        coef = np.polyfit(ys[sel], xs[sel], deg=1)
        trend = np.polyval(coef, ys).astype(np.float32)
    return trend


def _enforce_curve_consistency(centers: np.ndarray, x_positions: list[float]) -> list[float]:
    if len(x_positions) < 6:
        return x_positions

    arr = np.asarray(x_positions, np.float32)
    guide_mid = np.asarray(_median_smooth_positions(x_positions, radius=2), np.float32)
    guide_wide = np.asarray(_median_smooth_positions(x_positions, radius=5), np.float32)
    guide = 0.65 * guide_mid + 0.35 * guide_wide
    trend = _fit_tip_trend(centers, guide.tolist())
    if trend is None:
        trend = guide.copy()

    corrected = arr.copy()
    n = int(arr.size)
    top_count = max(6, int(round(0.40 * float(n))))
    for idx in range(n):
        pred = float(trend[idx])
        guide_x = float(guide[idx])
        x = float(arr[idx])
        right_dev = x - pred
        left_dev = pred - x
        guide_dev = x - guide_x
        is_top = idx < top_count

        if right_dev > 26.0 or abs(guide_dev) > 34.0 or (is_top and right_dev > 16.0):
            corrected[idx] = float(0.18 * x + 0.82 * pred)
        elif left_dev > 34.0:
            corrected[idx] = float(0.28 * x + 0.72 * pred)

    return _smooth_x_positions(corrected.tolist())


def _repair_low_confidence_positions(
    centers: np.ndarray,
    x_positions: list[float],
    confidences: list[float],
) -> list[float]:
    if len(x_positions) < 6 or len(confidences) != len(x_positions):
        return x_positions

    xs = np.asarray(x_positions, np.float32)
    conf = np.clip(np.asarray(confidences, np.float32), 0.0, 1.0)
    ys = np.asarray(centers, np.float32).reshape(-1)

    high = conf >= 0.55
    if int(np.count_nonzero(high)) < 6:
        high = conf >= 0.40
    if int(np.count_nonzero(high)) < 4:
        high = np.ones(xs.size, dtype=bool)

    try:
        coef = np.polyfit(ys[high], xs[high], deg=1)
        trend = np.polyval(coef, ys).astype(np.float32)
    except Exception:
        trend = np.asarray(_median_smooth_positions(x_positions, radius=3), np.float32)

    neigh = np.asarray(_median_smooth_positions(x_positions, radius=2), np.float32)
    corrected = xs.copy()
    top_count = max(6, int(round(0.45 * float(xs.size))))

    for idx in range(xs.size):
        x = float(xs[idx])
        c = float(conf[idx])
        pred = float(0.58 * trend[idx] + 0.42 * neigh[idx])
        dev = x - pred
        is_top = idx < top_count

        if c < 0.18:
            corrected[idx] = float(pred)
        elif c < 0.35 and abs(dev) > 8.0:
            corrected[idx] = float(0.10 * x + 0.90 * pred)
        elif c < 0.52:
            if is_top and dev > 6.0:
                corrected[idx] = float(0.18 * x + 0.82 * pred)
            elif abs(dev) > 11.0:
                corrected[idx] = float(0.24 * x + 0.76 * pred)

    return _smooth_x_positions(corrected.tolist())


def _repair_top_outliers_with_local_sam2(
    warp_bgr: np.ndarray,
    centers: np.ndarray,
    x_positions: list[float],
) -> list[float]:
    if warp_bgr is None or warp_bgr.size == 0 or len(x_positions) < 6:
        return x_positions

    ys = np.asarray(centers, np.float32).reshape(-1)
    xs = np.asarray(x_positions, np.float32)
    trend = _fit_tip_trend(ys, x_positions)
    if trend is None:
        trend = np.asarray(_median_smooth_positions(x_positions, radius=3), np.float32)

    corrected = xs.copy()
    h, w = warp_bgr.shape[:2]
    top_count = max(6, int(round(0.40 * float(xs.size))))
    for idx in range(min(top_count, xs.size)):
        x = float(xs[idx])
        pred = float(trend[idx])
        if x - pred <= 8.0:
            continue

        y = int(round(float(ys[idx])))
        lo, hi = _local_search_bounds(x, 0, w, idx, int(xs.size))
        x0 = max(0, int(round(lo - 40.0)))
        x1 = min(w, int(round(hi + 8.0)))
        y0 = max(0, y - 16)
        y1 = min(h, y + 16)
        crop = warp_bgr[y0:y1, x0:x1]
        candidate = None
        local_seg = _refine_tube_edge_with_local_felzenszwalb(crop)
        if local_seg is not None:
            local_x, local_score = local_seg
            seg_candidate = float(x0) + float(local_x)
            if pred - 20.0 <= seg_candidate <= x + 8.0 and local_score >= 0.35:
                candidate = seg_candidate

        if candidate is None:
            local = _refine_tube_edge_with_local_sam2(crop)
            if local is None:
                continue
            local_x, local_score = local
            sam_candidate = float(x0) + float(local_x)
            if sam_candidate < pred - 18.0:
                continue
            if sam_candidate > x + 8.0:
                continue
            if local_score < 0.02:
                continue
            candidate = sam_candidate

        corrected[idx] = float(0.18 * x + 0.82 * candidate)

    return _smooth_x_positions(corrected.tolist())


def _refine_x_positions_with_canny(
    gray_img: np.ndarray,
    centers: np.ndarray,
    base_x_positions: list[float],
    x_search0: int,
    x_search1: int,
    color_img: np.ndarray | None = None,
) -> tuple[list[float], list[float]]:
    if gray_img is None or gray_img.size == 0 or len(base_x_positions) == 0:
        return base_x_positions, [1.0 for _ in base_x_positions]

    guide_positions = _median_smooth_positions(_smooth_x_positions(base_x_positions), radius=2)
    tube_ref_lab = _estimate_tube_reference_lab(color_img, centers, guide_positions)
    lab_img = None
    if color_img is not None and color_img.size != 0:
        if color_img.ndim == 2:
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)
        else:
            color_bgr = color_img
        lab_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    pre_blur = cv2.GaussianBlur(gray_img, (1, 61), 0)
    grad_x = cv2.Sobel(pre_blur.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_x_pos = np.maximum(grad_x, 0.0)
    grad_x_neg = np.maximum(-grad_x, 0.0)
    edge_map = cv2.Canny(pre_blur.astype(np.uint8), 50, 150, L2gradient=True).astype(np.float32)

    refined: list[float] = []
    confidences: list[float] = []
    band_half = 4
    total_count = len(base_x_positions)
    for idx, (y_center, x_pred, x_guide) in enumerate(zip(centers.tolist(), base_x_positions, guide_positions)):
        y0 = max(0, int(y_center) - band_half)
        y1 = min(gray_img.shape[0] - 1, int(y_center) + band_half)
        strip_pos = grad_x_pos[y0 : y1 + 1]
        strip_neg = grad_x_neg[y0 : y1 + 1]
        strip_edge = edge_map[y0 : y1 + 1]
        if strip_pos.size == 0 or strip_neg.size == 0 or strip_edge.size == 0:
            refined.append(float(x_pred))
            confidences.append(0.10)
            continue

        prof_pos = _normalize_profile(np.percentile(strip_pos, 90, axis=0).astype(np.float32))
        prof_neg = _normalize_profile(np.percentile(strip_neg, 90, axis=0).astype(np.float32))
        prof_edge = _normalize_profile(cv2.GaussianBlur(np.mean(strip_edge, axis=0).astype(np.float32).reshape(1, -1), (1, 9), 0).reshape(-1))
        prof_pos_enh = np.power(prof_pos, 5.5)
        prof_neg_enh = np.power(prof_neg, 5.5)
        prof_edge_enh = np.power(prof_edge, 1.7)

        gray_band = gray_img[y0 : y1 + 1].astype(np.float32)
        prof_gray = _smooth_1d(np.percentile(gray_band, 60, axis=0).astype(np.float32), 11)

        x_seed = float(x_guide)
        lo, hi = _local_search_bounds(x_seed, x_search0, x_search1, idx, total_count)
        if hi <= lo + 3:
            refined.append(float(x_pred))
            confidences.append(0.10)
            continue

        search_pos = prof_pos_enh[lo:hi]
        search_neg = prof_neg_enh[lo:hi]
        search_edge = prof_edge_enh[lo:hi]
        if (
            search_pos.size == 0
            or search_neg.size == 0
            or search_edge.size == 0
            or max(float(np.max(search_pos)), float(np.max(search_neg)), float(np.max(search_edge))) < 0.03
        ):
            refined.append(float(x_pred))
            confidences.append(0.12)
            continue

        body_level = _profile_window_median(prof_gray, int(round(x_seed - 84.0)), int(round(x_seed - 34.0)), default=160.0)
        dark_level = _profile_window_median(prof_gray, int(round(x_seed + 8.0)), int(round(x_seed + 34.0)), default=max(32.0, body_level - 45.0))
        threshold_cross = _find_threshold_crossing(
            prof_gray,
            prof_neg_enh,
            lo,
            hi,
            body_level,
            dark_level,
        )

        neg_peaks = _find_local_peaks_1d(
            search_neg,
            threshold=max(0.06, 0.45 * float(np.max(search_neg))),
            min_distance=4,
        )
        pos_peaks = _find_local_peaks_1d(
            search_pos,
            threshold=max(0.06, 0.42 * float(np.max(search_pos))),
            min_distance=4,
        )

        best_score = -1e9
        best_x = None
        best_conf = 0.10
        x_pred_local = x_seed - float(lo)
        for neg_idx in neg_peaks:
            pos_candidates = [p for p in pos_peaks if 3 <= (p - neg_idx) <= 28]
            if not pos_candidates:
                continue
            neg_val = float(search_neg[neg_idx])
            if neg_val < 0.08:
                continue
            for pos_idx in pos_candidates:
                pos_val = float(search_pos[pos_idx])
                if pos_val < 0.05:
                    continue
                edge_slice = search_edge[max(0, neg_idx - 2) : min(search_edge.size, pos_idx + 3)]
                edge_val = float(np.max(edge_slice)) if edge_slice.size else 0.0
                gap = float(pos_idx - neg_idx)
                pred_penalty = abs(float(neg_idx) - x_pred_local)
                right_penalty = max(0.0, float(neg_idx) - x_pred_local - 6.0)
                candidate_x = float(lo + neg_idx)
                cand_i = int(round(candidate_x))
                left_far = _profile_window_median(prof_gray, cand_i - 56, cand_i - 26, default=body_level)
                left_near = _profile_window_median(prof_gray, cand_i - 22, cand_i - 6, default=body_level)
                right_near = _profile_window_median(prof_gray, cand_i + 4, cand_i + 18, default=dark_level)
                right_far = _profile_window_median(prof_gray, cand_i + 18, cand_i + 34, default=dark_level)
                drop_near = max(0.0, left_near - right_near)
                drop_far = max(0.0, left_far - right_far)
                body_match = max(0.0, 1.0 - abs(left_far - body_level) / max(16.0, 0.25 * body_level))
                transition_bonus = 0.0 if threshold_cross is None else max(0.0, 1.0 - abs(candidate_x - float(threshold_cross)) / 10.0)
                sample_lab = _sample_tube_body_lab(lab_img, float(y_center), candidate_x)
                color_dist = _lab_color_distance(sample_lab, tube_ref_lab)
                color_score = max(0.0, 1.0 - (color_dist / 18.0))
                color_penalty = max(0.0, (color_dist - 10.0) / 12.0)
                score = (
                    1.20 * neg_val
                    + 0.80 * pos_val
                    + 0.55 * edge_val
                    + 0.030 * drop_near
                    + 0.016 * drop_far
                    + 0.30 * body_match
                    + 0.38 * transition_bonus
                    + 0.22 * color_score
                    - 0.018 * pred_penalty
                    - 0.055 * right_penalty
                    - 0.012 * abs(gap - 11.0)
                    - 0.35 * color_penalty
                    - 0.004 * max(0.0, candidate_x - float(lo))
                )
                if score > best_score:
                    best_score = score
                    best_x = float(lo + neg_idx)
                    conf_val = (
                        0.38 * min(1.0, neg_val / 0.42)
                        + 0.16 * min(1.0, pos_val / 0.24)
                        + 0.18 * min(1.0, edge_val / 0.30)
                        + 0.14 * min(1.0, drop_near / 24.0)
                        + 0.08 * body_match
                        + 0.06 * transition_bonus
                    )
                    best_conf = float(np.clip(conf_val, 0.0, 1.0))

        if best_x is None:
            edge_combo = (0.72 * search_neg) + (0.20 * search_edge) + (0.08 * search_pos)
            peaks = _find_local_peaks_1d(
                edge_combo,
                threshold=max(0.05, 0.45 * float(np.max(edge_combo))),
                min_distance=4,
            )
            if peaks:
                best_local = None
                best_local_score = -1e9
                for local_idx in peaks:
                    candidate_x = float(lo + local_idx)
                    cand_i = int(round(candidate_x))
                    left_far = _profile_window_median(prof_gray, cand_i - 56, cand_i - 26, default=body_level)
                    left_near = _profile_window_median(prof_gray, cand_i - 22, cand_i - 6, default=body_level)
                    right_near = _profile_window_median(prof_gray, cand_i + 4, cand_i + 18, default=dark_level)
                    right_far = _profile_window_median(prof_gray, cand_i + 18, cand_i + 34, default=dark_level)
                    drop_near = max(0.0, left_near - right_near)
                    drop_far = max(0.0, left_far - right_far)
                    body_match = max(0.0, 1.0 - abs(left_far - body_level) / max(16.0, 0.25 * body_level))
                    transition_bonus = 0.0 if threshold_cross is None else max(0.0, 1.0 - abs(candidate_x - float(threshold_cross)) / 10.0)
                    sample_lab = _sample_tube_body_lab(lab_img, float(y_center), candidate_x)
                    color_dist = _lab_color_distance(sample_lab, tube_ref_lab)
                    color_score = max(0.0, 1.0 - (color_dist / 18.0))
                    color_penalty = max(0.0, (color_dist - 10.0) / 12.0)
                    score = (
                        float(edge_combo[local_idx])
                        + 0.022 * drop_near
                        + 0.010 * drop_far
                        + 0.24 * body_match
                        + 0.32 * transition_bonus
                        + 0.12 * color_score
                        - 0.018 * abs(float(local_idx) - x_pred_local)
                        - 0.25 * color_penalty
                        - 0.004 * max(0.0, candidate_x - float(lo))
                    )
                    if score > best_local_score:
                        best_local_score = score
                        best_local = local_idx
                        conf_val = (
                            0.32 * min(1.0, float(edge_combo[local_idx]) / 0.35)
                            + 0.18 * min(1.0, drop_near / 24.0)
                            + 0.10 * min(1.0, drop_far / 28.0)
                            + 0.18 * body_match
                            + 0.10 * transition_bonus
                            + 0.12 * color_score
                        )
                        best_conf = float(np.clip(conf_val, 0.0, 1.0))
                best_x = float(lo + int(best_local)) if best_local is not None else float(lo + int(peaks[0]))
            else:
                if threshold_cross is not None:
                    best_x = float(threshold_cross)
                    best_conf = 0.24
                else:
                    best_x = float(lo + int(np.argmax(edge_combo)))
                    best_conf = 0.16

        best_x = _pull_to_earliest_transition(
            best_x,
            threshold_cross,
            prof_gray,
            body_level,
            dark_level,
        )

        refined.append(best_x)
        confidences.append(float(np.clip(best_conf, 0.0, 1.0)))

    return _smooth_x_positions(refined), confidences


def _extend_centers_with_support(
    centers: list[int],
    period: float | None,
    gray_img: np.ndarray,
    x0: int,
    x1: int,
) -> list[int]:
    if not centers or period is None or not np.isfinite(period) or period < 8.0:
        return centers

    h, w = gray_img.shape[:2]
    sx0 = max(20, min(int(x0) + 28, w - 4))
    sx1 = max(sx0 + 24, min(int(x1) - 84, w))
    if sx1 <= sx0 + 24:
        sx0 = max(18, int(round(0.12 * float(w))))
        sx1 = min(w - 8, int(round(0.60 * float(w))))
    strip = gray_img[:, sx0:sx1]
    if strip.size == 0:
        return centers

    blur = cv2.GaussianBlur(strip, (15, 3), 0)
    darkness = np.maximum(0.0, 145.0 - blur.astype(np.float32))
    support = np.mean(darkness, axis=1)
    support_n = _normalize_01(_smooth_1d(support, 7))
    support_thr = 0.15
    half_band = max(4, int(round(0.22 * float(period))))
    max_extra = 4

    extended = [int(v) for v in centers]

    def band_score(center_y: int) -> float:
        y0 = max(0, int(center_y) - half_band)
        y1 = min(h - 1, int(center_y) + half_band)
        if y1 < y0:
            return 0.0
        return float(np.mean(support_n[y0 : y1 + 1]))

    added = 0
    while added < max_extra:
        predicted = int(round(extended[0] - float(period)))
        if predicted < 0:
            break
        if band_score(predicted) < support_thr:
            break
        extended.insert(0, predicted)
        added += 1

    added = 0
    while added < 2:
        predicted = int(round(extended[-1] + float(period)))
        if predicted >= h:
            break
        if band_score(predicted) < support_thr:
            break
        extended.append(predicted)
        added += 1

    return extended


def _find_best_periodic_peak_run(peaks: list[int], gap_lo: int = 14, gap_hi: int = 34, min_len: int = 6) -> tuple[list[int], float | None]:
    ordered = sorted(int(p) for p in peaks)
    if len(ordered) < min_len:
        return ordered, _median_period(ordered)

    best_run: list[int] = []
    best_period: float | None = None
    for start in range(len(ordered)):
        run = [ordered[start]]
        for current in ordered[start + 1 :]:
            gap = int(current - run[-1])
            if gap_lo <= gap <= gap_hi:
                run.append(current)
            elif gap > gap_hi:
                break
        if len(run) >= min_len:
            period = float(np.median(np.diff(np.asarray(run, np.float32))))
            if not best_run or len(run) > len(best_run) or (len(run) == len(best_run) and run[0] < best_run[0]):
                best_run = run
                best_period = period

    if best_run:
        return best_run, best_period
    return ordered, _median_period(ordered)


def _estimate_stack_band(
    warp_bgr: np.ndarray,
    preferred_x0: int,
    preferred_x1: int,
) -> tuple[int, int, list[int], float | None] | None:
    if warp_bgr is None or warp_bgr.size == 0:
        return None

    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY) if warp_bgr.ndim == 3 else warp_bgr.copy()
    h, w = gray.shape[:2]
    x0 = 0
    x1 = min(w - 8, max(54, int(round(0.14 * float(w)))))
    if x1 <= x0 + 20:
        x0 = 0
        x1 = min(w, max(32, int(round(0.18 * float(w)))))
    strip = gray[:, x0:x1]
    if strip.size == 0:
        return None

    blur_x = cv2.GaussianBlur(strip, (71, 1), 0)
    dark_profile = 255.0 - np.mean(blur_x.astype(np.float32), axis=1)
    profile_norm = _normalize_01(_smooth_1d(dark_profile, 9))
    if profile_norm.size == 0:
        return None

    peaks = _find_local_peaks_1d(profile_norm, threshold=max(0.32, float(np.percentile(profile_norm, 72))), min_distance=16)
    run, period = _find_best_periodic_peak_run(peaks, gap_lo=14, gap_hi=34, min_len=6)
    if len(run) < 6:
        return None

    period_used = period or _median_period(run) or 22.0
    y0 = max(0, int(round(run[0] - 0.9 * float(period_used))))
    y1 = min(h, int(round(run[-1] + 0.7 * float(period_used))))
    return (y0, y1, run, float(period_used))


def _extend_periodic_seams(
    base_run: list[int],
    observed_peaks: list[int],
    period: float | None,
    profile_norm: np.ndarray,
    y_min: int,
    y_max: int,
) -> list[int]:
    if not base_run:
        return []

    if period is None or not np.isfinite(period) or period < 8.0:
        return sorted(int(v) for v in base_run)

    observed = sorted(int(v) for v in observed_peaks)
    result = sorted(int(v) for v in base_run)
    snap_radius = max(5, int(round(0.40 * float(period))))
    lo_support = max(0.16, float(np.percentile(profile_norm, 55)))
    min_sep = max(10, int(round(0.55 * float(period))))

    def snap_or_predict(predicted: int) -> int | None:
        candidates = [peak for peak in observed if abs(int(peak) - predicted) <= snap_radius]
        if candidates:
            return int(min(candidates, key=lambda peak: abs(int(peak) - predicted)))
        if 0 <= predicted < profile_norm.size and float(profile_norm[int(predicted)]) >= lo_support:
            return int(predicted)
        return None

    while result:
        predicted = int(round(result[0] - float(period)))
        if predicted < y_min:
            break
        snapped = snap_or_predict(predicted)
        if snapped is None or (result[0] - snapped) < min_sep:
            break
        result.insert(0, int(snapped))

    while result:
        predicted = int(round(result[-1] + float(period)))
        if predicted > y_max:
            break
        snapped = snap_or_predict(predicted)
        if snapped is None or (snapped - result[-1]) < min_sep:
            break
        result.append(int(snapped))

    deduped: list[int] = []
    for peak in result:
        if not deduped or abs(int(peak) - deduped[-1]) >= min_sep:
            deduped.append(int(peak))
    return deduped


def _row_max_dark_run(mask_bool: np.ndarray) -> np.ndarray:
    runs: list[int] = []
    for row in np.asarray(mask_bool, bool):
        padded = np.r_[False, row, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        lengths = changes[1::2] - changes[::2]
        runs.append(int(lengths.max()) if lengths.size else 0)
    return np.asarray(runs, np.float32)


def _continuous_ranges(indices: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    values = np.asarray(indices, np.int32).reshape(-1)
    if values.size == 0:
        return []

    ranges: list[tuple[int, int]] = []
    start = prev = int(values[0])
    for raw in values[1:].tolist():
        value = int(raw)
        if value == prev + 1:
            prev = value
            continue
        if prev - start + 1 >= int(min_len):
            ranges.append((start, prev))
        start = prev = value

    if prev - start + 1 >= int(min_len):
        ranges.append((start, prev))
    return ranges


def _pick_cam152_tube_stack_range(gray_img: np.ndarray) -> tuple[int, int, int, int, int, np.ndarray]:
    dark_thr = int(round(np.percentile(gray_img, 38)))
    dark_thr = max(95, min(170, dark_thr))
    dark_mask = gray_img <= dark_thr

    run_profile = _row_max_dark_run(dark_mask)
    run_profile_smooth = _smooth_1d(run_profile, 11)
    min_stack_y = int(round(0.10 * float(gray_img.shape[0])))  # era 0.25
    # Usar percentil 25 del perfil como umbral adaptativo (no fijo 0.14×w que es demasiado alto)
    nonzero_runs = run_profile_smooth[run_profile_smooth > 10]
    if nonzero_runs.size >= 10:
        min_dark_run = max(50, int(round(float(np.percentile(nonzero_runs, 25)))))
    else:
        min_dark_run = max(50, int(round(0.06 * float(gray_img.shape[1]))))

    active = run_profile_smooth >= float(min_dark_run)
    active[:min_stack_y] = False
    ranges = _continuous_ranges(np.flatnonzero(active), min_len=8)
    if not ranges:
        return 0, gray_img.shape[0] - 1, 0, gray_img.shape[1], int(dark_thr), run_profile_smooth

    # El paquete completo va desde el primer rango activo hasta el último,
    # independientemente de separadores físicos intermedios.
    stack_y0 = ranges[0][0]
    stack_y1 = ranges[-1][1]

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
    with open("debug.txt", "a") as _f:
        _f.write(f"[pick_range] img={gray_img.shape} dark_thr={dark_thr} min_dark_run={min_dark_run} min_stack_y={min_stack_y} ranges={ranges} stack_y0={stack_y0} stack_y1={stack_y1}\n")
    return int(stack_y0), int(stack_y1), int(stack_x0), int(stack_x1), int(dark_thr), run_profile_smooth


def _is_cam152_image(image_path: str | Path) -> bool:
    stem = Path(image_path).stem.lower()
    return stem in {"cam_152", "cam152"} or stem.startswith("cam_152_") or stem.startswith("cam152_")


def _needs_backend_mirror(image_path: str | Path) -> bool:
    # cam_152.jpeg ya está espejada en disco — solo las variantes cam_152_* necesitan mirror.
    stem = Path(image_path).stem.lower()
    return stem.startswith("cam_152_") or stem.startswith("cam152_")


def _detect_tubes_in_warp(warp_bgr: np.ndarray, *, cam152_mode: bool = False) -> tuple[np.ndarray, dict]:
    with open("debug.txt", "a") as _f:
        _f.write(f"cam152_mode={cam152_mode} shape={warp_bgr.shape if warp_bgr is not None else None}\n")
    if warp_bgr is None or warp_bgr.size == 0:
        raise RuntimeError("warp image is empty")

    if warp_bgr.ndim == 2:
        gray = warp_bgr.copy()
        warp_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
        warp_vis = warp_bgr.copy()

    h, w = gray.shape[:2]

    if cam152_mode:
        tube_stack_y0, tube_stack_y1, strip_x0, strip_x1, _tube_dark_thr, _tube_run_profile = _pick_cam152_tube_stack_range(gray)
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
        if profile_norm.size == 0:
            raise RuntimeError("empty warp profile")

        energy_start_index = int(tube_stack_y0)
        profile_cut = profile_norm[tube_stack_y0 : tube_stack_y1 + 1]
        dominant_period, _autocorr = _estimate_dominant_period(profile_cut, min_period_frac=0.04, max_period_frac=0.14)

        expected_tube_pitch = float(dominant_period) if dominant_period is not None and np.isfinite(dominant_period) else 22.0
        pitch_lo = max(8.0, 0.60 * expected_tube_pitch)
        pitch_hi = max(pitch_lo + 1.0, 1.35 * expected_tube_pitch)
        # Dentro del paquete todo pico es un límite de tubo → threshold=0, min_distance mínimo físico.
        base_min_distance = max(6, int(round(0.30 * expected_tube_pitch)))

        peaks_index = [
            int(tube_stack_y0 + peak)
            for peak in _find_local_peaks_1d(profile_cut, threshold=0.0, min_distance=base_min_distance)
        ]

        gap_lo_seed = int(round(pitch_lo))
        gap_hi_seed = int(round(pitch_hi))
        run_peaks, run_period = _find_best_periodic_peak_run(peaks_index, gap_lo=gap_lo_seed, gap_hi=gap_hi_seed, min_len=6)
        median_period = _median_period(run_peaks or peaks_index)
        period_used = run_period or median_period or dominant_period or expected_tube_pitch or 22.0

        valid_observed_peaks = [int(p) for p in peaks_index if tube_stack_y0 <= int(p) <= tube_stack_y1]

        # Corregir período si _find_best_periodic_peak_run detectó 2× el período real
        # (picos alternos fuerte/débil cuando threshold=0 genera picos densos).
        if len(valid_observed_peaks) >= 4:
            all_gaps = np.diff(np.asarray(sorted(valid_observed_peaks), np.float32))
            median_all_gap = float(np.median(all_gaps))
            if period_used > 1.6 * median_all_gap and median_all_gap >= 8.0:
                period_used = median_all_gap

        # Usar todos los picos válidos (no solo run_peaks) ya que threshold=0 los acepta todos.
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
        tube_top_rows_list = []
        tube_bottom_rows_list = []
        rejected_tube_gaps: list[dict] = []
        for prev, curr in zip(peaks_filled[:-1], peaks_filled[1:]):
            gap = float(curr - prev)
            gap_in_range = lo_gap <= gap <= hi_gap
            both_inside_pack = (
                tube_stack_y0 <= float(prev) <= tube_stack_y1
                and tube_stack_y0 <= float(curr) <= tube_stack_y1
            )
            if gap_in_range or both_inside_pack:
                tube_top_rows_list.append(int(prev))
                tube_bottom_rows_list.append(int(curr))
            else:
                rejected_tube_gaps.append(
                    {
                        "from_y": int(prev),
                        "to_y": int(curr),
                        "distance_px": float(gap),
                        "reason": "alto" if gap > hi_gap else "bajo",
                    }
                )
    else:
        # The vertical tube-period profile is more stable near the left edge of
        # cam151 warps. Farther right, shadows and tube tips distort the seam periodicity.
        strip_x0 = 0
        strip_x1 = min(w - 8, max(54, int(round(0.14 * float(w)))))
        if strip_x1 <= strip_x0 + 20:
            strip_x0 = 0
            strip_x1 = min(w, max(32, int(round(0.18 * float(w)))))
        gray_strip = gray[:, strip_x0:strip_x1]

        blur_x = cv2.GaussianBlur(gray_strip, (71, 1), 0)
        dark_profile = 255.0 - np.mean(blur_x.astype(np.float32), axis=1)
        profile_smooth = _smooth_1d(dark_profile, 9)
        profile_norm = _normalize_01(profile_smooth)

        if profile_norm.size == 0:
            raise RuntimeError("empty warp profile")

        strong_rows = np.flatnonzero(profile_norm >= max(0.30, float(np.percentile(profile_norm, 70))))
        energy_start_index = int(strong_rows[0]) if strong_rows.size else 0
        profile_cut = profile_norm[energy_start_index:]
        dominant_period, _autocorr = _estimate_dominant_period(profile_cut, min_period_frac=0.03, max_period_frac=0.18)

        base_min_distance = 16
        if dominant_period is not None and np.isfinite(dominant_period):
            base_min_distance = max(12, int(round(0.65 * float(dominant_period))))
        min_distance_top = max(8, int(round(0.55 * float(base_min_distance))))
        min_distance_bottom = max(min_distance_top + 2, int(round(1.00 * float(base_min_distance))))
        base_threshold = max(0.32, float(np.percentile(profile_norm, 72)))

        first_peak_seed = None
        seed_period = float(dominant_period) if dominant_period is not None and np.isfinite(dominant_period) else float(base_min_distance)
        seed_lo = min(profile_norm.size - 2, max(1, energy_start_index + 2))
        seed_hi = min(profile_norm.size - 2, max(seed_lo + 2, energy_start_index + int(round(0.95 * seed_period))))
        first_threshold = min(base_threshold, max(0.18, 0.80 * float(base_threshold)))
        seed_candidates: list[int] = []
        for idx in range(seed_lo, seed_hi + 1):
            if profile_norm[idx] >= first_threshold and profile_norm[idx] >= profile_norm[idx - 1] and profile_norm[idx] > profile_norm[idx + 1]:
                seed_candidates.append(int(idx))
        if seed_candidates:
            first_peak_seed = int(seed_candidates[0])
        else:
            local_seed = profile_norm[seed_lo : seed_hi + 1]
            if local_seed.size:
                seed_arg = int(seed_lo + int(np.argmax(local_seed)))
                if float(profile_norm[seed_arg]) >= max(0.16, 0.70 * float(base_threshold)):
                    first_peak_seed = int(seed_arg)

        raw_peak_candidates: list[int] = []
        for idx in range(1, profile_norm.size - 1):
            if profile_norm[idx] >= base_threshold and profile_norm[idx] >= profile_norm[idx - 1] and profile_norm[idx] > profile_norm[idx + 1]:
                if idx >= energy_start_index:
                    raw_peak_candidates.append(int(idx))

        def _local_min_gap(pos_idx: int) -> int:
            denom = max(1.0, float(profile_norm.size - 1 - energy_start_index))
            rel = (float(pos_idx) - float(energy_start_index)) / denom
            rel = min(1.0, max(0.0, rel))
            gap = (1.0 - rel) * float(min_distance_top) + rel * float(min_distance_bottom)
            return max(1, int(round(gap)))

        peaks_index = []
        for cand in raw_peak_candidates:
            if not peaks_index:
                peaks_index.append(int(cand))
                continue
            gap_needed = _local_min_gap(cand)
            if int(cand) - int(peaks_index[-1]) >= gap_needed:
                peaks_index.append(int(cand))
            elif float(profile_norm[cand]) > float(profile_norm[peaks_index[-1]]):
                peaks_index[-1] = int(cand)

        if first_peak_seed is not None:
            if not peaks_index:
                peaks_index = [int(first_peak_seed)]
            else:
                first_gap_needed = max(4, int(round(0.35 * float(min_distance_top))))
                if int(first_peak_seed) < int(peaks_index[0]) and (int(peaks_index[0]) - int(first_peak_seed)) >= first_gap_needed:
                    peaks_index = [int(first_peak_seed)] + peaks_index

        run_peaks, run_period = _find_best_periodic_peak_run(peaks_index, gap_lo=14, gap_hi=34, min_len=6)
        median_period = _median_period(run_peaks or peaks_index)
        period_used = run_period or median_period or dominant_period or 22.0
        peaks_filled = _fill_missing_seams(run_peaks or peaks_index, period_used)
        peaks_filled = _extend_periodic_seams(
            peaks_filled,
            peaks_index,
            period_used,
            profile_norm,
            y_min=max(0, energy_start_index - int(round(0.6 * float(period_used)))),
            y_max=int(profile_norm.size - 1),
        )
        lo_gap = max(12.0, 0.55 * float(period_used))
        hi_gap = max(lo_gap + 1.0, 1.55 * float(period_used))

        tube_top_rows_list = []
        tube_bottom_rows_list = []
        rejected_tube_gaps = []
        if peaks_filled:
            top_gap = float(peaks_filled[0] - energy_start_index)
            if lo_gap <= top_gap <= hi_gap:
                tube_top_rows_list.append(int(energy_start_index))
                tube_bottom_rows_list.append(int(peaks_filled[0]))
        for prev, curr in zip(peaks_filled[:-1], peaks_filled[1:]):
            gap = float(curr - prev)
            if lo_gap <= gap <= hi_gap:
                tube_top_rows_list.append(int(prev))
                tube_bottom_rows_list.append(int(curr))

    tube_top_rows = np.asarray(tube_top_rows_list, np.int32)
    tube_bottom_rows = np.asarray(tube_bottom_rows_list, np.int32)

    blur_kernel = (21, 1)
    x_search0 = max(int(round(0.32 * float(w))), strip_x0)
    x_search1 = max(x_search0 + 8, min(w, int(round(0.98 * float(w)))))
    roi_keep_frac = 5.0 / 8.0

    raw_pick_x_positions: list[float] = []
    probe_rows_list: list[int] = []
    roi_boxes: list[tuple[int, int]] = []
    for y_top, y_bot in zip(tube_top_rows, tube_bottom_rows):
        y0_full = max(0, int(y_top))
        y1 = min(gray.shape[0] - 1, int(y_bot))
        band_h = max(1, int(y1 - y0_full + 1))
        roi_h = max(3, min(band_h, int(round(roi_keep_frac * float(band_h)))))
        y0 = max(y0_full, int(y1 - roi_h + 1))
        fallback_x = float(raw_pick_x_positions[-1]) if raw_pick_x_positions else float(x_search0)
        hit_y = int(round(0.5 * float(y0 + y1)))
        x_pick = float(fallback_x)
        roi_boxes.append((int(y0), int(y1)))

        roi_gray = gray[y0 : y1 + 1, x_search0:x_search1]
        if roi_gray.size == 0:
            raw_pick_x_positions.append(float(x_pick))
            probe_rows_list.append(int(hit_y))
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
        profile_power = 12.0 if cam152_mode else 2.0
        profile_1d = (
            np.power(_normalize_01(_smooth_1d(profile_raw, 9)), profile_power)
            if profile_raw.size
            else np.empty((0,), np.float32)
        )

        x_local = None
        if profile_1d.size and float(np.max(profile_1d)) > 1e-6:
            peak_threshold = max(0.18, 0.45 * float(np.max(profile_1d)))
            cand_x = _find_local_peaks_1d(profile_1d, threshold=peak_threshold, min_distance=5)
            if cand_x:
                x_local = int(min(cand_x))
            else:
                x_local = int(np.argmax(profile_1d))

        if x_local is not None:
            neg_strength = float(np.max(roi_grad_x_neg[:, x_local])) if roi_grad_x_neg.shape[0] > 0 else 0.0
            pos_strength = float(np.max(roi_grad_x_pos[:, x_local])) if roi_grad_x_pos.shape[0] > 0 else 0.0
            if neg_strength >= max(1e-6, 0.75 * pos_strength):
                edge_col = roi_grad_x_neg[:, x_local].astype(np.float32)
            else:
                edge_col = roi_grad_x_pos[:, x_local].astype(np.float32)

            edge_col = edge_col - float(np.min(edge_col))
            edge_den = float(np.max(edge_col))
            if edge_den > 1e-6:
                edge_col = edge_col / edge_den
            else:
                edge_col = np.zeros_like(edge_col, np.float32)

            cand_rows = np.flatnonzero(edge_col >= max(0.30, 0.70 * float(np.max(edge_col))))
            ref_rel_y = None
            if probe_rows_list:
                ref_rel_y = float(np.median(np.asarray(probe_rows_list[-3:], np.float32))) - float(y0)

            if cand_rows.size == 0:
                hit_local_y = int(np.argmax(edge_col)) if edge_col.size else int(round(0.5 * float(y1 - y0)))
            else:
                best_y_score = -1e9
                hit_local_y = int(cand_rows[0])
                for cand_y in cand_rows.tolist():
                    top_bonus = 1.0 - (float(cand_y) / max(1.0, float(edge_col.size - 1)))
                    consistency_y = 0.0
                    if ref_rel_y is not None:
                        consistency_y = max(
                            0.0,
                            1.0 - abs(float(cand_y) - ref_rel_y) / max(4.0, 0.35 * float(edge_col.size)),
                        )
                    y_score = (1.00 * top_bonus) + (0.55 * consistency_y) + (0.25 * float(edge_col[cand_y]))
                    if y_score > best_y_score:
                        best_y_score = y_score
                        hit_local_y = int(cand_y)

            hit_y = int(y0 + hit_local_y)
            x_pick = float(x_search0 + x_local)

        raw_pick_x_positions.append(float(x_pick))
        probe_rows_list.append(int(hit_y))

    probe_rows = np.asarray(probe_rows_list, np.int32)

    if cam152_mode:
        roi_overlay_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(roi_overlay_vis, (x_search0, 0), (x_search1 - 1, h - 1), (96, 255, 96), 1)
        x_start_list: list[dict] = []
        tube_count_for_numbering = min(len(roi_boxes), len(raw_pick_x_positions), len(probe_rows))
        for idx, ((y0, y1), x_final, y_probe) in enumerate(zip(roi_boxes, raw_pick_x_positions, probe_rows), start=1):
            tube_number = max(1, tube_count_for_numbering - idx + 1)
            y0i = max(0, int(y0))
            y1i = min(gray.shape[0] - 1, int(y1))
            x_i = int(round(float(x_final)))
            y_probe_i = int(round(float(y_probe)))
            cv2.rectangle(roi_overlay_vis, (x_search0, y0i), (x_search1 - 1, y1i), (0, 220, 255), 1)
            cv2.line(roi_overlay_vis, (x_search0, y0i), (x_search1 - 1, y0i), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(roi_overlay_vis, (x_i, y1i), (x_i, y_probe_i), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(roi_overlay_vis, (x_i, y_probe_i), 2, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                roi_overlay_vis,
                str(tube_number),
                (x_search0 + 4, max(12, y0i + 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            x_start_list.append(
                {
                    "tube_idx": int(tube_number),
                    "x_start": float(x_final),
                    "x_local": float(x_final),
                    "x_end_estimate": float(x_final),
                    "x_seed": float(x_final),
                    "y_center": float(y_probe),
                    "confidence": 1.0,
                }
            )

        x_start_list.sort(key=lambda item: int(item["tube_idx"]))

        return roi_overlay_vis, {
            "tube_count": int(len(x_start_list)),
            "dominant_period": float(period_used) if period_used is not None else None,
            "energy_start_index": int(energy_start_index),
            "peaks_index": [int(v) for v in peaks_index],
            "peaks_index_dom": [int(v) for v in peaks_filled],
            "peaks_index_filtered": [int(v) for v in peaks_filled],
            "cluster_start_idx": None,
            "x_start_list": x_start_list,
            "processing_mode": "cam152",
            "processing_stage": "cam152_mirrored_warp_roi_sobel_x_no_zoom",
            "pitch_lo": float(lo_gap),
            "pitch_hi": float(hi_gap),
            "rejected_tube_gaps": list(rejected_tube_gaps),
            "sam2_used": False,
            "sam2_score": None,
        }

    zoom_blur_kernel = (1, 101)
    zoom_half_width = 30
    zoom_min_width = 28
    zoom_half_height_min = 14

    zoom_seed_x_positions = list(raw_pick_x_positions)
    refine_x_positions: list[float] = []
    local_confidences: list[float] = []
    zoom_roi_boxes: list[tuple[int, int, int, int]] = []
    for (y0, y1), x_seed, y_hit in zip(roi_boxes, zoom_seed_x_positions, probe_rows):
        y0_band = max(0, int(y0))
        y1_band = min(gray.shape[0] - 1, int(y1))
        band_h = max(1, int(y1_band - y0_band + 1))
        zoom_half_height = max(zoom_half_height_min, int(round(0.35 * float(band_h))))
        hit_i = int(round(float(y_hit)))
        y0i = max(y0_band, hit_i - zoom_half_height)
        y1i = min(y1_band, hit_i + zoom_half_height)
        if y1i <= y0i + 8:
            y0i = y0_band
            y1i = y1_band

        seed_i = int(round(float(x_seed)))
        zx0 = int(round(float(seed_i) - float(zoom_half_width)))
        zx1 = int(round(float(seed_i) + float(zoom_half_width) + 1.0))
        if zx0 < int(x_search0):
            shift = int(x_search0) - zx0
            zx0 += shift
            zx1 += shift
        if zx1 > int(x_search1):
            shift = zx1 - int(x_search1)
            zx0 -= shift
            zx1 -= shift
        zx0 = max(int(x_search0), zx0)
        zx1 = min(int(x_search1), zx1)
        if zx1 <= zx0 + zoom_min_width:
            deficit = zoom_min_width - (zx1 - zx0)
            zx0 = max(int(x_search0), zx0 - int(round(0.50 * float(deficit))))
            zx1 = min(int(x_search1), zx1 + int(round(0.50 * float(deficit))))
        if zx1 <= zx0 + 8:
            refine_x_positions.append(float(x_seed))
            local_confidences.append(0.0)
            zoom_roi_boxes.append((y0i, y1i, zx0, zx1))
            continue

        roi_gray_zoom = gray[y0i : y1i + 1, zx0:zx1]
        if roi_gray_zoom.size == 0:
            refine_x_positions.append(float(x_seed))
            local_confidences.append(0.0)
            zoom_roi_boxes.append((y0i, y1i, zx0, zx1))
            continue

        roi_blur_zoom = cv2.GaussianBlur(roi_gray_zoom, zoom_blur_kernel, 0)
        roi_grad_x_zoom = cv2.Sobel(roi_blur_zoom.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        roi_sobel_abs_zoom = np.abs(roi_grad_x_zoom).astype(np.float32)
        roi_sobel_abs_zoom = roi_sobel_abs_zoom - float(np.min(roi_sobel_abs_zoom))
        sobel_den = float(np.max(roi_sobel_abs_zoom))
        if sobel_den > 1e-6:
            roi_sobel_abs_zoom = roi_sobel_abs_zoom / sobel_den
        else:
            roi_sobel_abs_zoom = np.zeros_like(roi_sobel_abs_zoom, np.float32)

        profile_raw = np.max(roi_sobel_abs_zoom, axis=0).astype(np.float32)
        profile_power = 12.0 if cam152_mode else 2.0
        profile_1d = (
            np.power(_normalize_01(_smooth_1d(profile_raw, 9)), profile_power)
            if profile_raw.size
            else np.empty((0,), np.float32)
        )

        x_local = None
        conf_val = 0.0
        if profile_1d.size and float(np.max(profile_1d)) > 1e-6:
            peak_threshold = max(0.16, 0.55 * float(np.max(profile_1d)))
            cand_x = _find_local_peaks_1d(profile_1d, threshold=peak_threshold, min_distance=3)
            if cand_x:
                x_local = int(min(cand_x))
            else:
                x_local = int(np.argmax(profile_1d))
            conf_val = float(profile_1d[x_local])

        x_pick = float(x_seed) if x_local is None else float(zx0 + x_local)
        refine_x_positions.append(float(x_pick))
        local_confidences.append(float(conf_val))
        zoom_roi_boxes.append((y0i, y1i, zx0, zx1))

    # Cam152 keeps the Sobel-X ROI seed as the final point. The zoom pass is
    # diagnostic only for that camera because it can pull points too far right.
    final_x_positions = list(zoom_seed_x_positions if cam152_mode else refine_x_positions)
    final_x_positions = [
        float(min(max(x_val, float(zx0)), float(max(zx0, zx1 - 1))))
        for x_val, (_y0i, _y1i, zx0, zx1) in zip(final_x_positions, zoom_roi_boxes)
    ]
    x_start_list: list[dict] = []
    tube_count_for_numbering = min(
        len(zoom_roi_boxes),
        len(zoom_seed_x_positions),
        len(refine_x_positions),
        len(final_x_positions),
        len(probe_rows),
        len(local_confidences),
    )
    for idx, ((y0i, y1i, zx0, zx1), x_seed, x_zoom, x_final, y_probe, conf_val) in enumerate(
        zip(
            zoom_roi_boxes,
            zoom_seed_x_positions,
            refine_x_positions,
            final_x_positions,
            probe_rows,
            local_confidences,
        ),
        start=1,
    ):
        tube_number = max(1, tube_count_for_numbering - idx + 1)
        cv2.rectangle(warp_vis, (int(zx0), int(y0i)), (int(zx1 - 1), int(y1i)), (0, 220, 255), 1)
        cv2.line(warp_vis, (int(zx0), int(y0i)), (int(zx1 - 1), int(y0i)), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(warp_vis, (int(round(x_final)), int(y0i)), (int(round(x_final)), int(y1i)), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(warp_vis, (int(round(x_final)), int(round(y_probe))), 2, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            warp_vis,
            str(tube_number),
            (int(zx0) + 4, max(12, int(y0i) + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        x_start_list.append(
            {
                "tube_idx": int(tube_number),
                "x_start": float(x_final),
                "x_local": float(x_zoom),
                "x_end_estimate": float(x_final),
                "x_seed": float(x_seed),
                "y_center": float(y_probe),
                "confidence": float(conf_val),
            }
        )

    x_start_list.sort(key=lambda item: int(item["tube_idx"]))

    for y in peaks_index:
        cv2.line(warp_vis, (0, int(y)), (warp_vis.shape[1] - 1, int(y)), (0, 180, 255), 1, cv2.LINE_AA)
    for y in peaks_filled:
        cv2.line(warp_vis, (0, int(y)), (warp_vis.shape[1] - 1, int(y)), (0, 255, 120), 1, cv2.LINE_AA)

    cv2.rectangle(warp_vis, (strip_x0, 0), (strip_x1 - 1, warp_vis.shape[0] - 1), (96, 255, 96), 1)
    cv2.rectangle(warp_vis, (x_search0, 0), (x_search1 - 1, warp_vis.shape[0] - 1), (64, 200, 255), 1)
    cv2.line(warp_vis, (0, int(energy_start_index)), (warp_vis.shape[1] - 1, int(energy_start_index)), (255, 255, 255), 2, cv2.LINE_AA)
    info = f"tubes={len(x_start_list)} start={energy_start_index}"
    if period_used is not None:
        info += f" period={float(period_used):.1f}px"
    cv2.putText(warp_vis, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    return warp_vis, {
        "tube_count": int(len(x_start_list)),
        "dominant_period": float(period_used) if period_used is not None else None,
        "energy_start_index": int(energy_start_index),
        "peaks_index": [int(v) for v in peaks_index],
        "peaks_index_dom": [int(v) for v in peaks_filled],
        "peaks_index_filtered": [int(v) for v in peaks_filled],
        "cluster_start_idx": None,
        "x_start_list": x_start_list,
        "processing_mode": "cam152" if cam152_mode else "cam151",
        "processing_stage": "cam151_zoom_refine" if not cam152_mode else "cam152_mirrored_warp_roi_sobel_x_no_zoom",
        "pitch_lo": float(lo_gap),
        "pitch_hi": float(hi_gap),
        "rejected_tube_gaps": list(rejected_tube_gaps),
        "sam2_used": False,
        "sam2_score": None,
    }


def _normalize_roi(roi: dict) -> dict:
    xyxy = roi.get("xyxy") or [0, 0, 0, 0]
    return {
        "id": str(roi.get("id") or ""),
        "note": str(roi.get("note") or ""),
        "xyxy": [int(round(float(v))) for v in xyxy[:4]],
    }


def _mirror_roi_source_x(roi: dict | None, source_width: int) -> dict | None:
    if roi is None or source_width <= 0:
        return roi
    mirrored = dict(roi)
    x1, y1, x2, y2 = [int(round(float(v))) for v in roi.get("xyxy", [0, 0, 0, 0])[:4]]
    mx1 = int(source_width - 1 - x2)
    mx2 = int(source_width - 1 - x1)
    mirrored["xyxy"] = [min(mx1, mx2), y1, max(mx1, mx2), y2]
    return mirrored


def _mirror_line_source_x(line: dict, source_width: int) -> dict:
    if source_width <= 0 or _line_is_warp_space(line):
        return line
    mirrored = dict(line)
    mirrored["x1"] = int(source_width - 1 - int(round(float(line.get("x1", 0)))))
    mirrored["x2"] = int(source_width - 1 - int(round(float(line.get("x2", 0)))))
    return mirrored


def _pick_detection_roi(rois: list[dict]) -> dict | None:
    if not rois:
        return None

    normalized = [_normalize_roi(roi) for roi in rois]
    if not normalized:
        return None

    def rank(roi: dict) -> tuple[int, float, float]:
        note = roi["note"].strip().lower()
        lower_bonus = 1 if any(token in note for token in ("lower", "bottom", "tube", "tubos")) else 0
        x1, y1, x2, y2 = roi["xyxy"]
        cy = 0.5 * (y1 + y2)
        area = max(1, x2 - x1) * max(1, y2 - y1)
        return (lower_bonus, cy, area)

    normalized.sort(key=rank, reverse=True)
    return normalized[0]


def _warp_detection_roi(
    roi: dict | None,
    src_points: list[tuple[float, float]],
    output_size: tuple[int, int],
    warp_shape: tuple[int, int, int] | tuple[int, int],
    *,
    output_flip_horizontal: bool = False,
) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None

    width, height = int(output_size[0]), int(output_size[1])
    if width <= 1 or height <= 1:
        return None

    src = np.float32(src_points)
    dst = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    transform = cv2.getPerspectiveTransform(src, dst)
    if output_flip_horizontal:
        output_flip = np.float32(
            [
                [-1.0, 0.0, float(width - 1)],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = output_flip @ transform

    x1, y1, x2, y2 = roi["xyxy"]
    roi_pts = np.float32([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
    warped = cv2.perspectiveTransform(roi_pts, transform).reshape(-1, 2)
    if warped.size == 0 or not np.all(np.isfinite(warped)):
        return None

    wx0 = int(np.floor(float(np.min(warped[:, 0]))))
    wy0 = int(np.floor(float(np.min(warped[:, 1]))))
    wx1 = int(np.ceil(float(np.max(warped[:, 0]))))
    wy1 = int(np.ceil(float(np.max(warped[:, 1]))))

    h, w = warp_shape[:2]
    raw_h = max(2, wy1 - wy0)
    expand_up = max(220, int(round(0.28 * float(h))), int(round(8.0 * float(raw_h))))
    expand_down = max(16, int(round(1.5 * float(raw_h))))
    expand_x = 18

    wx0 = max(0, min(wx0 - expand_x, w - 2))
    wx1 = w
    wy0 = max(0, min(wy0 - expand_up, h - 2))
    wy1 = max(wy0 + 2, min(wy1 + expand_down, h))
    return (wx0, wy0, wx1, wy1)


def build_tube_detection_preview(
    image_path: str | Path,
    lines: list[dict],
    points: list[dict],
    rois: list[dict],
    output_dir: str | Path,
    *,
    src_points_override: list | None = None,
    dst_rect_override: list | tuple | dict | None = None,
) -> TubeDetectionPreviewResult:
    image_path = Path(image_path)
    cam152_mode = _is_cam152_image(image_path)
    mirror_in_backend = _needs_backend_mirror(image_path)
    homography = build_homography_preview(
        image_path=image_path,
        lines=lines,
        points=points,
        output_dir=output_dir,
        src_points_override=src_points_override,
        dst_rect_override=dst_rect_override,
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
    if warp_bgr is None:
        raise FileNotFoundError(f"Cannot read warp preview: {homography.warp_path}")

    selected_roi = _pick_detection_roi(rois)
    if mirror_in_backend and homography.source_size[0] > 0:
        selected_roi = _mirror_roi_source_x(selected_roi, homography.source_size[0])
    detection_roi = _warp_detection_roi(
        selected_roi,
        homography.src_points,
        homography.output_size,
        warp_bgr.shape,
        output_flip_horizontal=mirror_in_backend,
    )
    if detection_roi is not None and not cam152_mode:
        stack_band = _estimate_stack_band(warp_bgr, detection_roi[0], detection_roi[2])
        if stack_band is not None:
            band_y0, band_y1, _run, _period = stack_band
            top_expand = max(220, int(round(13.0 * float(_period or 22.0))))
            detection_roi = (
                int(detection_roi[0]),
                int(min(detection_roi[1], max(0, band_y0 - top_expand))),
                int(detection_roi[2]),
                int(max(detection_roi[3], band_y1)),
            )

    with open("debug.txt", "a") as _f:
        _f.write(f"detection_roi={detection_roi} cam152_mode={cam152_mode} warp_shape={warp_bgr.shape}\n")
    if detection_roi is None:
        overlay_bgr, summary = _detect_tubes_in_warp(warp_bgr, cam152_mode=cam152_mode)
    else:
        x0, y0, x1, y1 = detection_roi
        warp_crop = warp_bgr[y0:y1, x0:x1]
        crop_overlay, summary = _detect_tubes_in_warp(warp_crop, cam152_mode=cam152_mode)
        overlay_bgr = warp_bgr.copy()
        overlay_bgr[y0:y1, x0:x1] = crop_overlay
        cv2.rectangle(overlay_bgr, (x0 + 2, y0 + 2), (min(x1 - 2, x0 + 220), min(y1 - 2, y0 + 34)), (12, 18, 26), -1)
        cv2.rectangle(overlay_bgr, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            overlay_bgr,
            "tube ROI",
            (x0 + 8, max(24, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        summary["energy_start_index"] = int(summary["energy_start_index"]) + y0
        summary["peaks_index"] = [int(v) + y0 for v in summary["peaks_index"]]
        summary["peaks_index_dom"] = [int(v) + y0 for v in summary["peaks_index_dom"]]
        if summary.get("peaks_index_filtered") is not None:
            summary["peaks_index_filtered"] = [int(v) + y0 for v in summary["peaks_index_filtered"]]
        if summary.get("rejected_tube_gaps") is not None:
            adjusted_rejected_gaps: list[dict] = []
            for gap in summary["rejected_tube_gaps"]:
                gap_item = dict(gap)
                if gap_item.get("from_y") is not None:
                    gap_item["from_y"] = int(gap_item["from_y"]) + y0
                if gap_item.get("to_y") is not None:
                    gap_item["to_y"] = int(gap_item["to_y"]) + y0
                adjusted_rejected_gaps.append(gap_item)
            summary["rejected_tube_gaps"] = adjusted_rejected_gaps
        adjusted_x_starts: list[dict] = []
        for item in summary["x_start_list"]:
            adjusted_item = dict(item)
            adjusted_item["tube_idx"] = int(item["tube_idx"])
            for key in ("x_start", "x_local", "x_end_estimate", "x_seed"):
                if adjusted_item.get(key) is not None:
                    adjusted_item[key] = float(adjusted_item[key]) + x0
            if adjusted_item.get("y_center") is not None:
                adjusted_item["y_center"] = float(adjusted_item["y_center"]) + y0
            adjusted_x_starts.append(adjusted_item)
        summary["x_start_list"] = adjusted_x_starts
        info = f"tubes={len(summary['x_start_list'])} roi_y={y0}:{y1}"
        if summary["dominant_period"] is not None:
            info += f" period={float(summary['dominant_period']):.1f}px"
        cv2.putText(
            overlay_bgr,
            info,
            (x0 + 8, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if mirror_in_backend and homography.source_size[0] > 0:
        src_w = homography.source_size[0]
        points_for_references = [dict(p, x=int(src_w - 1 - int(p.get("x", 0)))) for p in points]
        lines_for_references = [_mirror_line_source_x(line, src_w) for line in lines]
    else:
        points_for_references = points
        lines_for_references = lines

    px_per_in, reference_lines, scale_samples = _compute_reference_measurements(
        points_for_references, final_transform.tolist(), lines=lines_for_references, cam152_mode=cam152_mode
    )
    reference_source_lines = _compute_reference_source_lines(points_for_references, cam152_mode=cam152_mode)
    source_overlay_bgr = cv2.imread(str(homography.overlay_path), cv2.IMREAD_COLOR)
    source_overlay_path = Path(output_dir) / "tube_detection_overlay.jpg"
    source_overlay_dirty = False
    if source_overlay_bgr is not None and reference_source_lines:
        for idx, ref in enumerate(reference_source_lines, start=1):
            color = (255, 210, 80) if idx == 1 else (80, 220, 255)
            top_pt = ref["top_point"]
            bottom_pt = ref["bottom_point"]
            cv2.line(
                source_overlay_bgr,
                (int(round(float(top_pt[0]))), int(round(float(top_pt[1])))),
                (int(round(float(bottom_pt[0]))), int(round(float(bottom_pt[1])))),
                color,
                3,
                cv2.LINE_AA,
            )
            cv2.circle(
                source_overlay_bgr,
                (int(round(float(top_pt[0]))), int(round(float(top_pt[1])))),
                5,
                color,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                source_overlay_bgr,
                (int(round(float(bottom_pt[0]))), int(round(float(bottom_pt[1])))),
                5,
                color,
                -1,
                cv2.LINE_AA,
            )
            cv2.putText(
                source_overlay_bgr,
                f"{ref['label']} {ref.get('top_label', '')}->{ref.get('bottom_label', '')}",
                (int(round(float(top_pt[0]))) + 8, max(26, int(round(float(top_pt[1]))) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
        source_overlay_dirty = True

    if reference_lines:
        for idx, ref in enumerate(reference_lines, start=1):
            x_top = _line_x_at_y(ref["top_point"], ref["bottom_point"], 0.0)
            x_bottom = _line_x_at_y(ref["top_point"], ref["bottom_point"], float(overlay_bgr.shape[0] - 1))
            color = (255, 210, 80) if idx == 1 else (80, 220, 255)
            cv2.line(
                overlay_bgr,
                (int(round(x_top)), 0),
                (int(round(x_bottom)), overlay_bgr.shape[0] - 1),
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                overlay_bgr,
                (int(round(float(ref["top_point"][0]))), int(round(float(ref["top_point"][1])))),
                4,
                color,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                overlay_bgr,
                (int(round(float(ref["bottom_point"][0]))), int(round(float(ref["bottom_point"][1])))),
                4,
                color,
                -1,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay_bgr,
                f"{ref['label']} {ref.get('top_label', '')}->{ref.get('bottom_label', '')}",
                (min(overlay_bgr.shape[1] - 180, int(round(x_top)) + 6), 22 + 18 * (idx - 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

    if px_per_in is not None:
        cv2.putText(
            overlay_bgr,
            f"scale={px_per_in:.2f}px/in",
            (12, min(overlay_bgr.shape[0] - 12, 52)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if reference_lines:
        for item in summary["x_start_list"]:
            x_val = float(item.get("x_start", item.get("x_local", 0.0)))
            y_val = float(item.get("y_center", 0.0))
            ref_distances: dict[str, dict] = {}
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

    ref2 = next((ref for ref in reference_lines if str(ref.get("label")) == "ref_02"), None)
    if ref2 is None and reference_lines:
        ref2 = reference_lines[-1]  # fallback: usar la ultima referencia disponible (p.ej. ref_01 en cam152)
    if ref2 is not None:
        measure_color = (0, 200, 255)
        for item in summary["x_start_list"]:
            x_val = float(item.get("x_start", item.get("x_local", 0.0)))
            y_val = float(item.get("y_center", 0.0))
            ref_x = _line_x_at_y(ref2["top_point"], ref2["bottom_point"], y_val)
            x0_line = int(round(min(x_val, ref_x)))
            x1_line = int(round(max(x_val, ref_x)))
            y_line = int(round(y_val))
            if x1_line <= x0_line + 1:
                continue

            cv2.line(
                overlay_bgr,
                (x0_line, y_line),
                (x1_line, y_line),
                measure_color,
                1,
                cv2.LINE_AA,
            )

            if source_overlay_bgr is not None:
                tube_src = _project_xy(final_inverse_transform, x_val, y_val)
                ref_src = _project_xy(final_inverse_transform, ref_x, y_val)
                if tube_src is not None and ref_src is not None:
                    sx0, sy0 = tube_src
                    sx1, sy1 = ref_src
                    src_measure_color = (0, 60, 255)
                    cv2.line(
                        source_overlay_bgr,
                        (int(round(sx0)), int(round(sy0))),
                        (int(round(sx1)), int(round(sy1))),
                        src_measure_color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.circle(source_overlay_bgr, (int(round(sx0)), int(round(sy0))), 3, src_measure_color, -1, cv2.LINE_AA)
                    cv2.circle(source_overlay_bgr, (int(round(sx1)), int(round(sy1))), 3, src_measure_color, -1, cv2.LINE_AA)
                    source_overlay_dirty = True

            dist_info = dict(item.get("ref_distances", {})).get(str(ref2.get("label", "ref_02")), {})
            if dist_info.get("distance_in") is not None:
                measure_text = f'{float(dist_info["distance_in"]):.1f}"'
            else:
                measure_text = f'{float(abs(x_val - ref_x)):.0f}px'

            text_x = int(round(0.5 * float(x0_line + x1_line)))
            text_y = max(14, y_line - 4)
            (tw, th), baseline = cv2.getTextSize(measure_text, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            box_x0 = max(0, text_x - 2)
            box_y0 = max(0, text_y - th - 3)
            box_x1 = min(overlay_bgr.shape[1] - 1, box_x0 + tw + 4)
            box_y1 = min(overlay_bgr.shape[0] - 1, text_y + baseline)
            cv2.rectangle(overlay_bgr, (box_x0, box_y0), (box_x1, box_y1), (18, 18, 18), -1)
            cv2.putText(
                overlay_bgr,
                measure_text,
                (box_x0 + 2, max(th + 1, text_y - 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                measure_color,
                1,
                cv2.LINE_AA,
            )

            if source_overlay_bgr is not None and tube_src is not None and ref_src is not None:
                text_src_x = int(round(0.5 * float(tube_src[0] + ref_src[0])))
                text_src_y = max(14, int(round(0.5 * float(tube_src[1] + ref_src[1]))) - 3)
                (stw, sth), sb = cv2.getTextSize(measure_text, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
                sbox_x0 = max(0, text_src_x - 2)
                sbox_y0 = max(0, text_src_y - sth - 3)
                sbox_x1 = min(source_overlay_bgr.shape[1] - 1, sbox_x0 + stw + 4)
                sbox_y1 = min(source_overlay_bgr.shape[0] - 1, text_src_y + sb)
                cv2.rectangle(source_overlay_bgr, (sbox_x0, sbox_y0), (sbox_x1, sbox_y1), (18, 18, 18), -1)
                cv2.putText(
                    source_overlay_bgr,
                    measure_text,
                    (sbox_x0 + 2, max(sth + 1, text_src_y - 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 60, 255),
                    1,
                    cv2.LINE_AA,
                )
                source_overlay_dirty = True

    if source_overlay_bgr is not None and source_overlay_dirty:
        cv2.imwrite(str(source_overlay_path), source_overlay_bgr)
        homography.overlay_path = source_overlay_path

    detection_overlay_path = Path(output_dir) / "tube_detection_warp.jpg"
    cv2.imwrite(str(detection_overlay_path), overlay_bgr)

    return TubeDetectionPreviewResult(
        homography=homography,
        detection_overlay_path=detection_overlay_path,
        tube_count=int(summary["tube_count"]),
        dominant_period=summary["dominant_period"],
        energy_start_index=int(summary["energy_start_index"]),
        peaks_index=list(summary["peaks_index"]),
        peaks_index_dom=list(summary["peaks_index_dom"]),
        x_start_list=list(summary["x_start_list"]),
        detection_roi=detection_roi,
        px_per_in=px_per_in,
        reference_lines=reference_lines,
        scale_samples=scale_samples,
        processing_mode=str(summary.get("processing_mode") or ("cam152" if cam152_mode else "cam151")),
        processing_stage=str(summary.get("processing_stage") or ("cam152_mirrored_warp_roi_sobel_x_no_zoom" if cam152_mode else "cam151_zoom_refine")),
        pitch_lo=float(summary["pitch_lo"]) if summary.get("pitch_lo") is not None else None,
        pitch_hi=float(summary["pitch_hi"]) if summary.get("pitch_hi") is not None else None,
        rejected_tube_gaps=list(summary.get("rejected_tube_gaps") or []),
    )
