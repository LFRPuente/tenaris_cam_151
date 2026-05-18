from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


SCRIPT_DIR = Path(__file__).resolve().parent
PIPE_END_ROOT = SCRIPT_DIR.parent
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO box height versus vertical position.")
    parser.add_argument("--images-root", type=Path, default=PIPE_END_ROOT / "annotation_pool" / "images")
    parser.add_argument("--labels-root", type=Path, default=PIPE_END_ROOT / "annotation_pool" / "labels")
    parser.add_argument("--camera", default="cam151")
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "pipe_end_yolo_analysis")
    parser.add_argument("--outlier-z", type=float, default=2.5)
    return parser.parse_args()


def image_size(path: Path) -> tuple[int, int]:
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None and img.size:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass
    raise ValueError(f"Could not read image size: {path}")


def find_image(images_root: Path, rel_stem: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = (images_root / rel_stem).with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def parse_label_file(label_path: Path, image_path: Path, images_root: Path) -> list[dict]:
    width, height = image_size(image_path)
    rows: list[dict] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id != 0:
            continue
        x_center_n, y_center_n, box_width_n, box_height_n = [float(value) for value in parts[1:5]]
        confidence = float(parts[5]) if len(parts) >= 6 else None
        rows.append(
            {
                "image": image_path.relative_to(images_root).as_posix(),
                "label": label_path.as_posix(),
                "line": line_number,
                "image_width": width,
                "image_height": height,
                "x_center_norm": x_center_n,
                "y_center_norm": y_center_n,
                "box_width_norm": box_width_n,
                "box_height_norm": box_height_n,
                "x_center_px": x_center_n * width,
                "y_center_px": y_center_n * height,
                "box_width_px": box_width_n * width,
                "box_height_px": box_height_n * height,
                "confidence": confidence,
            }
        )
    return rows


def collect_boxes(images_root: Path, labels_root: Path, camera: str) -> list[dict]:
    camera_labels = labels_root / camera
    camera_images = images_root / camera
    if not camera_labels.exists():
        raise FileNotFoundError(camera_labels)
    if not camera_images.exists():
        raise FileNotFoundError(camera_images)

    rows: list[dict] = []
    for label_path in sorted(camera_labels.glob("*.txt")):
        if not label_path.exists() or not label_path.read_text(encoding="utf-8").strip():
            continue
        rel_stem = Path(camera) / label_path.with_suffix("").name
        image_path = find_image(images_root, rel_stem)
        if image_path is None:
            continue
        rows.extend(parse_label_file(label_path, image_path, images_root))
    return rows


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return None
    return num / (den_x * den_y)


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-12:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    intercept = my - slope * mx
    return intercept, slope


def summarize_bins(rows: list[dict], bins: int) -> list[dict]:
    bins = max(1, int(bins))
    summary: list[dict] = []
    for idx in range(bins):
        y0 = idx / bins
        y1 = (idx + 1) / bins
        bucket = [
            row
            for row in rows
            if y0 <= float(row["y_center_norm"]) < y1 or (idx == bins - 1 and float(row["y_center_norm"]) <= y1)
        ]
        heights = [float(row["box_height_px"]) for row in bucket]
        summary.append(
            {
                "bin": idx + 1,
                "y_norm_min": y0,
                "y_norm_max": y1,
                "count": len(bucket),
                "height_px_median": median(heights) if heights else None,
                "height_px_mean": sum(heights) / len(heights) if heights else None,
                "height_px_min": min(heights) if heights else None,
                "height_px_max": max(heights) if heights else None,
            }
        )
    return summary


def enrich_with_fit(rows: list[dict], intercept: float, slope: float, outlier_z: float) -> list[dict]:
    residuals: list[float] = []
    for row in rows:
        expected = intercept + slope * float(row["y_center_norm"])
        row["expected_height_px"] = expected
        row["height_residual_px"] = float(row["box_height_px"]) - expected
        row["height_ratio_to_expected"] = float(row["box_height_px"]) / expected if expected > 1e-6 else None
        residuals.append(abs(float(row["height_residual_px"])))

    residual_median = median(residuals) if residuals else 0.0
    mad = median([abs(value - residual_median) for value in residuals]) if residuals else 0.0
    robust_scale = max(1e-6, 1.4826 * mad)
    outliers = []
    for row in rows:
        z_score = abs(float(row["height_residual_px"])) / robust_scale
        row["height_residual_robust_z"] = z_score
        if z_score >= float(outlier_z):
            outliers.append(row)
    return sorted(outliers, key=lambda row: float(row["height_residual_robust_z"]), reverse=True)


def write_plot(rows: list[dict], bins: list[dict], out_path: Path, intercept: float, slope: float) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    xs = [float(row["y_center_norm"]) for row in rows]
    ys = [float(row["box_height_px"]) for row in rows]
    plt.figure(figsize=(9, 5))
    plt.scatter(xs, ys, s=10, alpha=0.35, label="boxes")
    line_x = [0.0, 1.0]
    line_y = [intercept + slope * x for x in line_x]
    plt.plot(line_x, line_y, color="#d62728", linewidth=2, label="linear fit")
    bin_x = [0.5 * (float(row["y_norm_min"]) + float(row["y_norm_max"])) for row in bins if row["height_px_median"] is not None]
    bin_y = [float(row["height_px_median"]) for row in bins if row["height_px_median"] is not None]
    if bin_x:
        plt.plot(bin_x, bin_y, color="#1f77b4", marker="o", linewidth=2, label="bin median")
    plt.xlabel("Y center normalized, 0=top, 1=bottom")
    plt.ylabel("Box height, px")
    plt.title("cam151 pipe_end box height by vertical position")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    rows = collect_boxes(args.images_root.resolve(), args.labels_root.resolve(), args.camera)
    if len(rows) < 2:
        raise SystemExit(f"Not enough boxes for analysis: {len(rows)}")

    xs = [float(row["y_center_norm"]) for row in rows]
    ys = [float(row["box_height_px"]) for row in rows]
    intercept, slope = linear_fit(xs, ys)
    corr = pearson(xs, ys)
    outliers = enrich_with_fit(rows, intercept, slope, args.outlier_z)
    bins = summarize_bins(rows, args.bins)

    image_count = len({row["image"] for row in rows})
    payload = {
        "camera": args.camera,
        "image_count": image_count,
        "box_count": len(rows),
        "fit": {
            "height_px_intercept_at_top": intercept,
            "height_px_slope_per_full_image_y": slope,
            "pearson_y_norm_vs_height_px": corr,
            "expected_direction": "negative_slope_means_boxes_are_taller_toward_top",
        },
        "bins": bins,
        "outlier_z_threshold": float(args.outlier_z),
        "outlier_count": len(outliers),
        "top_outliers": outliers[:30],
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.camera}_box_height_by_y.json"
    csv_path = out_dir / f"{args.camera}_box_height_by_y.csv"
    plot_path = out_dir / f"{args.camera}_box_height_by_y.png"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    plot_written = write_plot(rows, bins, plot_path, intercept, slope)

    print(f"camera: {args.camera}")
    print(f"images: {image_count}")
    print(f"boxes: {len(rows)}")
    print(f"height_px_fit: {intercept:.3f} + ({slope:.3f} * y_norm)")
    print(f"pearson_y_vs_height: {corr:.4f}" if corr is not None else "pearson_y_vs_height: n/a")
    print(f"outliers_z>={args.outlier_z}: {len(outliers)}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    if plot_written:
        print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
