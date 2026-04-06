from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import Cam151Config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_bgr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def build_green_mask(image_bgr: np.ndarray, cfg: Cam151Config) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg.green_h_lo, cfg.green_s_lo, cfg.green_v_lo], dtype=np.uint8)
    upper = np.array([cfg.green_h_hi, cfg.green_s_hi, cfg.green_v_hi], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.green_open_size, cfg.green_open_size))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.green_close_size, cfg.green_close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    return mask


def extract_green_components(mask: np.ndarray, cfg: Cam151Config) -> list[dict]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[dict] = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label].tolist()
        if int(area) < int(cfg.min_green_area):
            continue
        cx, cy = centroids[label].tolist()
        aspect = float(h) / float(max(1, w))
        orientation = "vertical" if h >= w else "horizontal"
        components.append(
            {
                "label": int(label),
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
                "centroid": [float(cx), float(cy)],
                "aspect": float(aspect),
                "orientation": orientation,
            }
        )

    components.sort(key=lambda item: item["area"], reverse=True)
    return components


def draw_green_components(image_bgr: np.ndarray, components: list[dict]) -> np.ndarray:
    vis = image_bgr.copy()
    for idx, comp in enumerate(components, start=1):
        x1, y1, x2, y2 = comp["bbox"]
        color = (0, 255, 0) if comp["orientation"] == "vertical" else (0, 200, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"#{idx} {comp['orientation']} a={comp['area']}"
        cv2.putText(vis, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


def select_target_green_component(components: list[dict]) -> dict | None:
    verticals = [comp for comp in components if comp["orientation"] == "vertical"]
    if not verticals:
        return None
    return max(verticals, key=lambda item: (item["centroid"][0], item["area"]))


def draw_target_green_component(image_bgr: np.ndarray, component: dict, roi_rect: tuple[int, int, int, int]) -> np.ndarray:
    vis = image_bgr.copy()
    x1, y1, x2, y2 = component["bbox"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(vis, "target_green_support", (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    rx1, ry1, rx2, ry2 = roi_rect
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 150, 0), 2)
    cv2.putText(vis, "upper_posts_roi", (rx1 + 8, max(24, ry1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2, cv2.LINE_AA)
    return vis


def build_upper_posts_roi(image_bgr: np.ndarray, component: dict, cfg: Cam151Config) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, _y2 = component["bbox"]
    rx1 = max(0, int(x1) - int(cfg.target_green_x_margin_left))
    rx2 = min(width, int(x2) + int(cfg.target_green_x_margin_right))
    ry1 = max(0, int(y1) - int(cfg.target_green_y_margin_top))
    ry2 = min(height, int(y1) + int(cfg.target_green_y_margin_bottom))
    return image_bgr[ry1:ry2, rx1:rx2].copy(), (rx1, ry1, rx2, ry2)


def detect_upper_vertical_posts(
    image_bgr: np.ndarray,
    target_green_component: dict | None,
    cfg: Cam151Config,
) -> tuple[list[dict], tuple[int, int, int, int] | None, np.ndarray | None, np.ndarray | None]:
    if target_green_component is None:
        return [], None, None, None

    roi_bgr, roi_rect = build_upper_posts_roi(image_bgr, target_green_component, cfg)
    rx1, ry1, _rx2, _ry2 = roi_rect
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    thresholds = [
        ((cfg.post_red_h_lo_1, cfg.post_s_lo, cfg.post_v_lo), (cfg.post_red_h_hi_1, 255, 255)),
        ((cfg.post_red_h_lo_2, cfg.post_s_lo, cfg.post_v_lo), (cfg.post_red_h_hi_2, 255, 255)),
        ((cfg.post_orange_h_lo, cfg.post_s_lo, cfg.post_v_lo), (cfg.post_orange_h_hi, 255, 255)),
    ]
    for lo, hi in thresholds:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.post_open_size, cfg.post_open_size))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.post_close_width, cfg.post_close_height))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    posts: list[dict] = []
    vis = roi_bgr.copy()

    green_x1, green_y1, green_x2, _green_y2 = target_green_component["bbox"]
    green_center_x = 0.5 * (green_x1 + green_x2)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label].tolist()
        aspect = float(h) / float(max(1, w))
        if int(area) < int(cfg.post_min_area):
            continue
        if int(h) < int(cfg.post_min_height):
            continue
        if float(aspect) < float(cfg.post_min_aspect):
            continue

        cx, cy = centroids[label].tolist()
        gx1, gy1, gx2, gy2 = rx1 + x, ry1 + y, rx1 + x + w, ry1 + y + h
        gcx, gcy = rx1 + float(cx), ry1 + float(cy)
        posts.append(
            {
                "bbox": [int(gx1), int(gy1), int(gx2), int(gy2)],
                "center": [float(gcx), float(gcy)],
                "area": int(area),
                "width": int(w),
                "height": int(h),
                "aspect": float(aspect),
                "relative_to_green": {
                    "dx_from_green_center": float(gcx - green_center_x),
                    "dx_from_green_left": float(gcx - green_x1),
                    "dy_from_green_top": float(gcy - green_y1),
                },
            }
        )
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(vis, f"a={area}", (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    posts.sort(key=lambda item: item["center"][0])
    return posts, roi_rect, mask, vis


def bottom_roi(image_bgr: np.ndarray, cfg: Cam151Config) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image_bgr.shape[:2]
    y1 = int(round(height * cfg.bottom_roi_y_frac))
    y1 = max(0, min(y1, height - 1))
    roi = image_bgr[y1:height, 0:width].copy()
    return roi, (0, y1, width, height)


def detect_bottom_lines(roi_bgr: np.ndarray, cfg: Cam151Config) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_length,
        maxLineGap=cfg.hough_max_line_gap,
    )

    vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    line_items: list[dict] = []
    if lines is not None:
        for raw in lines[:, 0]:
            x1, y1, x2, y2 = [int(v) for v in raw.tolist()]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = float((dx * dx + dy * dy) ** 0.5)
            angle_deg = float(np.degrees(np.arctan2(dy, dx)))
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
            line_items.append(
                {
                    "p1": [x1, y1],
                    "p2": [x2, y2],
                    "length": length,
                    "angle_deg": angle_deg,
                }
            )

    line_items.sort(key=lambda item: item["length"], reverse=True)
    return edges, vis, line_items


def draw_bottom_roi(image_bgr: np.ndarray, roi_rect: tuple[int, int, int, int]) -> np.ndarray:
    vis = image_bgr.copy()
    x1, y1, x2, y2 = roi_rect
    cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (255, 200, 0), 2)
    cv2.putText(vis, "bottom_roi", (x1 + 8, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
    return vis


def run_cam151_bootstrap(image_path: Path, output_dir: Path, cfg: Cam151Config | None = None) -> dict:
    cfg = cfg or Cam151Config()
    ensure_dir(output_dir)

    image_bgr = load_bgr(image_path)
    green_mask = build_green_mask(image_bgr, cfg)
    green_components = extract_green_components(green_mask, cfg)
    green_vis = draw_green_components(image_bgr, green_components)
    target_green_component = select_target_green_component(green_components)
    upper_posts, upper_posts_roi_rect, upper_posts_mask, upper_posts_vis = detect_upper_vertical_posts(
        image_bgr,
        target_green_component,
        cfg,
    )

    roi_bgr, roi_rect = bottom_roi(image_bgr, cfg)
    roi_vis = draw_bottom_roi(image_bgr, roi_rect)
    bottom_edges, bottom_lines_vis, bottom_lines = detect_bottom_lines(roi_bgr, cfg)

    files = {
        "green_mask": output_dir / "01_green_mask.png",
        "green_components": output_dir / "02_green_components.jpg",
        "target_green_component": output_dir / "03_target_green_component.jpg",
        "upper_posts_mask": output_dir / "04_upper_posts_mask.png",
        "upper_posts": output_dir / "05_upper_vertical_posts.jpg",
        "bottom_roi": output_dir / "06_bottom_roi.jpg",
        "bottom_edges": output_dir / "07_bottom_edges.png",
        "bottom_lines": output_dir / "08_bottom_lines.jpg",
        "summary": output_dir / "summary.json",
    }

    cv2.imwrite(str(files["green_mask"]), green_mask)
    cv2.imwrite(str(files["green_components"]), green_vis)
    if target_green_component is not None and upper_posts_roi_rect is not None:
        target_green_vis = draw_target_green_component(image_bgr, target_green_component, upper_posts_roi_rect)
        cv2.imwrite(str(files["target_green_component"]), target_green_vis)
    if upper_posts_mask is not None:
        cv2.imwrite(str(files["upper_posts_mask"]), upper_posts_mask)
    if upper_posts_vis is not None:
        cv2.imwrite(str(files["upper_posts"]), upper_posts_vis)
    cv2.imwrite(str(files["bottom_roi"]), roi_vis)
    cv2.imwrite(str(files["bottom_edges"]), bottom_edges)
    cv2.imwrite(str(files["bottom_lines"]), bottom_lines_vis)

    summary = {
        "image_path": str(image_path),
        "image_shape": [int(v) for v in image_bgr.shape],
        "config": asdict(cfg),
        "green_components": green_components,
        "target_green_component": target_green_component,
        "upper_posts": {
            "roi_rect_xyxy": [int(v) for v in upper_posts_roi_rect] if upper_posts_roi_rect is not None else None,
            "count": len(upper_posts),
            "posts": upper_posts,
        },
        "bottom_roi": {
            "rect_xyxy": [int(v) for v in roi_rect],
            "shape": [int(v) for v in roi_bgr.shape],
        },
        "bottom_lines": bottom_lines[:50],
        "output_files": {key: str(path) for key, path in files.items() if key != "summary"},
    }

    files["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
