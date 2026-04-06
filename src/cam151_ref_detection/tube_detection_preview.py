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


@lru_cache(maxsize=1)
def _get_sam2_predictor(model_id: str = "facebook/sam2.1-hiera-tiny"):
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SAM2ImagePredictor.from_pretrained(model_id, device=device)


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
    predictor.set_image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

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

    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbs,
        box=box,
        multimask_output=True,
    )
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


def _refine_tube_edge_with_local_felzenszwalb(crop_bgr: np.ndarray) -> tuple[float, float] | None:
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 40:
        return None

    from skimage.measure import regionprops
    from skimage.segmentation import felzenszwalb

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


def _detect_tubes_in_warp(warp_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    if warp_bgr is None or warp_bgr.size == 0:
        raise RuntimeError("warp image is empty")

    if warp_bgr.ndim == 2:
        gray = warp_bgr.copy()
        warp_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
        warp_vis = warp_bgr.copy()

    h, w = gray.shape[:2]
    sam2_result = None
    sam2_mask = None

    # The vertical tube-period profile is more stable near the left edge of the
    # warp. Farther right, shadows and tube tips distort the seam periodicity.
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

    min_distance = 16
    if dominant_period is not None and np.isfinite(dominant_period):
        min_distance = max(12, int(round(0.65 * float(dominant_period))))
    base_threshold = max(0.32, float(np.percentile(profile_norm, 72)))
    peaks_index = _find_local_peaks_1d(profile_norm, threshold=base_threshold, min_distance=min_distance)
    peaks_index = [int(p) for p in peaks_index if p >= energy_start_index]

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
    gaps = np.diff(np.asarray(peaks_filled, np.int32)) if len(peaks_filled) >= 2 else np.empty((0,), np.int32)
    lo_gap = max(12.0, 0.55 * float(period_used))
    hi_gap = max(lo_gap + 1.0, 1.55 * float(period_used))

    centers_list: list[int] = []
    edge_top_list: list[int] = []
    edge_bot_list: list[int] = []
    for prev, curr in zip(peaks_filled[:-1], peaks_filled[1:]):
        gap = float(curr - prev)
        if lo_gap <= gap <= hi_gap:
            edge_top_list.append(int(prev))
            edge_bot_list.append(int(curr))
            centers_list.append(int(round(0.5 * (float(prev) + float(curr)))))

    edge_top = np.asarray(edge_top_list, np.int32)
    edge_bot = np.asarray(edge_bot_list, np.int32)
    centers = np.asarray(
        _extend_centers_with_support(
            centers_list,
            period_used,
            gray,
            strip_x0,
            strip_x1,
        ),
        np.int32,
    )

    light = gray
    tube_band_half = 4
    draw_band_half = 2
    tube_box_left = 34
    tube_box_right = 14
    # Strong vertical blur + signed Sobel X, closer to the old notebook logic.
    pre_blur = cv2.GaussianBlur(light, (1, 61), 0)
    grad_x = cv2.Sobel(pre_blur.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_x_pos = np.maximum(grad_x, 0.0)
    grad_x_neg = np.maximum(-grad_x, 0.0)
    x_search0 = max(int(round(0.48 * float(w))), strip_x0)
    x_search1 = max(x_search0 + 8, min(w, int(round(0.98 * float(w)))))

    x_start_list: list[dict] = []
    raw_x_positions: list[float] = []
    for idx, y_center in enumerate(centers, start=1):
        y0 = max(0, int(y_center) - tube_band_half)
        y1 = min(warp_vis.shape[0] - 1, int(y_center) + tube_band_half)
        strip_pos = grad_x_pos[y0 : y1 + 1, x_search0:x_search1]
        strip_neg = grad_x_neg[y0 : y1 + 1, x_search0:x_search1]
        if strip_pos.size == 0 or strip_neg.size == 0:
            continue

        prof_pos_raw = np.percentile(strip_pos, 90, axis=0).astype(np.float32)
        prof_neg_raw = np.percentile(strip_neg, 90, axis=0).astype(np.float32)
        if prof_pos_raw.size == 0 or prof_neg_raw.size == 0:
            continue

        exp_power = 6.0
        prof_pos = np.power(_normalize_profile(prof_pos_raw), exp_power)
        prof_neg = np.power(_normalize_profile(prof_neg_raw), exp_power)

        neg_max = float(np.max(prof_neg)) if prof_neg.size else 0.0
        pos_max = float(np.max(prof_pos)) if prof_pos.size else 0.0
        if max(neg_max, pos_max) <= 1e-6:
            continue

        neg_idx = int(np.argmax(prof_neg))
        pos_idx = int(np.argmax(prof_pos))

        # Tube start is the left edge of the shadow/body cutoff, so prefer the
        # strongest negative transition. Fall back to positive when the negative
        # edge is too weak or inconsistent.
        if neg_max >= max(0.18, 0.55 * pos_max):
            x_sel = int(x_search0 + neg_idx)
        else:
            x_sel = int(x_search0 + pos_idx)
        raw_x_positions.append(float(x_sel))

    local_x_positions = _smooth_x_positions(raw_x_positions)
    local_x_positions, local_confidences = _refine_x_positions_with_canny(
        gray,
        centers,
        local_x_positions,
        x_search0,
        x_search1,
        warp_bgr,
    )
    local_x_positions = _enforce_curve_consistency(centers, local_x_positions)
    local_x_positions = _repair_low_confidence_positions(centers, local_x_positions, local_confidences)
    local_x_positions = _repair_top_outliers_with_local_sam2(
        warp_bgr if warp_bgr.ndim == 3 else warp_vis,
        centers,
        local_x_positions,
    )
    smooth_x_positions = _stabilize_x_positions(centers, local_x_positions)
    for idx, (y_center, x_local, x_sel, conf_val) in enumerate(
        zip(centers, local_x_positions, smooth_x_positions, local_confidences),
        start=1,
    ):
        y0 = max(0, int(y_center) - draw_band_half)
        y1 = min(warp_vis.shape[0] - 1, int(y_center) + draw_band_half)
        x_local_i = int(round(x_local))
        x_i = int(round(x_sel))
        rx0 = max(0, x_local_i - tube_box_left)
        rx1 = min(warp_vis.shape[1] - 1, x_local_i + tube_box_right)
        low_conf = float(conf_val) < 0.35
        box_color = (0, 200, 255) if low_conf else (255, 255, 0)
        line_color = (0, 0, 255) if low_conf else (255, 0, 255)
        x_start_list.append(
            {
                "tube_idx": int(idx),
                "x_start": float(x_sel),
                "x_local": float(x_local),
                "x_end_estimate": float(x_local),
                "y_center": float(y_center),
                "confidence": float(conf_val),
            }
        )
        cv2.rectangle(warp_vis, (rx0, y0), (rx1, y1), box_color, 1)
        cv2.line(warp_vis, (x_i, y0), (x_i, y1), line_color, 1, cv2.LINE_AA)
        cv2.circle(warp_vis, (x_local_i, int(round(y_center))), 2, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            warp_vis,
            str(idx),
            (min(warp_vis.shape[1] - 24, x_local_i + 6), max(18, y0 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    for y in peaks_index:
        cv2.line(warp_vis, (0, int(y)), (warp_vis.shape[1] - 1, int(y)), (0, 180, 255), 1, cv2.LINE_AA)
    for y in peaks_filled:
        cv2.line(warp_vis, (0, int(y)), (warp_vis.shape[1] - 1, int(y)), (0, 255, 120), 1, cv2.LINE_AA)

    cv2.rectangle(warp_vis, (strip_x0, 0), (strip_x1 - 1, warp_vis.shape[0] - 1), (96, 255, 96), 1)
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
        "sam2_used": bool(sam2_mask is not None),
        "sam2_score": None if sam2_result is None else float(sam2_result["sam_score"]),
    }


def _normalize_roi(roi: dict) -> dict:
    xyxy = roi.get("xyxy") or [0, 0, 0, 0]
    return {
        "id": str(roi.get("id") or ""),
        "note": str(roi.get("note") or ""),
        "xyxy": [int(round(float(v))) for v in xyxy[:4]],
    }


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
) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None

    width, height = int(output_size[0]), int(output_size[1])
    if width <= 1 or height <= 1:
        return None

    src = np.float32(src_points)
    dst = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    transform = cv2.getPerspectiveTransform(src, dst)

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
    dst_rect_override: list | tuple | dict | None = None,
) -> TubeDetectionPreviewResult:
    homography = build_homography_preview(
        image_path=image_path,
        lines=lines,
        points=points,
        output_dir=output_dir,
        dst_rect_override=dst_rect_override,
    )

    warp_bgr = cv2.imread(str(homography.warp_path), cv2.IMREAD_COLOR)
    if warp_bgr is None:
        raise FileNotFoundError(f"Cannot read warp preview: {homography.warp_path}")

    detection_roi = _warp_detection_roi(
        _pick_detection_roi(rois),
        homography.src_points,
        homography.output_size,
        warp_bgr.shape,
    )
    if detection_roi is not None:
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

    if detection_roi is None:
        overlay_bgr, summary = _detect_tubes_in_warp(warp_bgr)
    else:
        x0, y0, x1, y1 = detection_roi
        warp_crop = warp_bgr[y0:y1, x0:x1]
        crop_overlay, summary = _detect_tubes_in_warp(warp_crop)
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
        adjusted_x_starts: list[dict] = []
        for item in summary["x_start_list"]:
            adjusted_x_starts.append(
                {
                    "tube_idx": int(item["tube_idx"]),
                    "x_start": float(item["x_start"]) + x0,
                    "x_local": float(item.get("x_local", item["x_start"])) + x0,
                    "y_center": float(item["y_center"]) + y0,
                }
            )
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
    )
