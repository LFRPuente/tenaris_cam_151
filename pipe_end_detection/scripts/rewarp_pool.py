"""Re-warp all raw cam151 captures with the current TOML ROI and auto-correct YOLO labels."""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURES_DIR = REPO_ROOT / "pipe_end_detection" / "captures" / "runs"
ANNOT_ROOT = REPO_ROOT / "pipe_end_detection" / "annotation_pool"
TOML_PATH = REPO_ROOT / "manual_rois" / "_current_defaults" / "cam_151_current_default_rois.toml"

# Old dst_rect (before the ROI picker change)
OLD_RECT = [-61.333, -157.988, 1120.232, 1335.930]


def dist(a: list, b: list) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


def warp_image(raw_path: Path, src_pts: list, dst_rect: list) -> np.ndarray:
    img = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(raw_path)

    tl, tr, bl, br = src_pts
    bw = max(240, int(round(max(dist(tl, tr), dist(bl, br)))))
    bh = max(320, int(round(max(dist(tl, bl), dist(tr, br)))))

    src = np.float32(src_pts)
    dst_base = np.float32([[0, 0], [bw-1, 0], [0, bh-1], [bw-1, bh-1]])
    M_base = cv2.getPerspectiveTransform(src, dst_base)
    M_inv  = np.linalg.inv(M_base)

    x0, y0, x1, y1 = dst_rect
    corners_d = np.float32([[[x0, y0]], [[x1, y0]], [[x0, y1]], [[x1, y1]]])
    corners_s = cv2.perspectiveTransform(corners_d, M_inv).reshape(-1, 2)

    ow = max(80, int(round(x1 - x0)))
    oh = max(80, int(round(y1 - y0)))
    out_corners = np.float32([[0, 0], [ow-1, 0], [0, oh-1], [ow-1, oh-1]])
    M_final = cv2.getPerspectiveTransform(corners_s, out_corners)

    return cv2.warpPerspective(img, M_final, (ow, oh))


def correct_labels(label_path: Path, old_rect: list, new_rect: list) -> str:
    """Translate YOLO label coords from old_rect space to new_rect space."""
    old_w = old_rect[2] - old_rect[0]
    old_h = old_rect[3] - old_rect[1]
    new_w = new_rect[2] - new_rect[0]
    new_h = new_rect[3] - new_rect[1]
    # pixel shift: same homo point → new pixel = old pixel + (old_offset - new_offset)
    dx = old_rect[0] - new_rect[0]  # shift in x pixels
    dy = old_rect[1] - new_rect[1]  # shift in y pixels

    lines_out: list[str] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            lines_out.append(line)
            continue
        cls = parts[0]
        xc_old = float(parts[1]) * old_w
        yc_old = float(parts[2]) * old_h
        w_old  = float(parts[3]) * old_w
        h_old  = float(parts[4]) * old_h

        xc_new = (xc_old + dx) / new_w
        yc_new = (yc_old + dy) / new_h
        w_new  = w_old / new_w
        h_new  = h_old / new_h

        # Clamp to [0,1]
        xc_new = max(0.0, min(1.0, xc_new))
        yc_new = max(0.0, min(1.0, yc_new))
        w_new  = max(0.0, min(1.0, w_new))
        h_new  = max(0.0, min(1.0, h_new))

        conf = f" {parts[5]}" if len(parts) == 6 else ""
        lines_out.append(f"{cls} {xc_new:.6f} {yc_new:.6f} {w_new:.6f} {h_new:.6f}{conf}")

    return "\n".join(lines_out) + ("\n" if lines_out else "")


def main() -> None:
    data = tomllib.loads(TOML_PATH.read_text(encoding="utf-8"))
    src_pts  = data["src_points_override"]
    new_rect = data["dst_rect_override"]

    print(f"TOML new dst_rect: {[round(v,3) for v in new_rect]}")
    print(f"Old dst_rect:      {[round(v,3) for v in OLD_RECT]}")
    print()

    raw_images = sorted(CAPTURES_DIR.glob("*/raw/cam_151_*.jpg"))
    print(f"Raw captures encontrados: {len(raw_images)}")

    ok = rewarped = skipped = label_fixed = 0

    for raw_path in raw_images:
        run_id   = raw_path.stem.replace("cam_151_", "")
        pool_img = ANNOT_ROOT / "images" / "cam151" / f"cam151_{run_id}.jpg"
        label_f  = ANNOT_ROOT / "labels" / "cam151" / f"cam151_{run_id}.txt"

        if not pool_img.exists():
            skipped += 1
            continue

        try:
            warped = warp_image(raw_path, src_pts, new_rect)
            cv2.imwrite(str(pool_img), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rewarped += 1

            if label_f.exists():
                content = label_f.read_text(encoding="utf-8").strip()
                if content:
                    corrected = correct_labels(label_f, OLD_RECT, new_rect)
                    label_f.write_text(corrected, encoding="utf-8")
                    label_fix_count = len([l for l in corrected.splitlines() if l.strip()])
                    label_fixed += 1
                    print(f"  ✓ {pool_img.name}  ({label_fix_count} boxes corregidos)")
                else:
                    print(f"  ✓ {pool_img.name}  (sin label)")
            else:
                print(f"  ✓ {pool_img.name}  (sin label)")
            ok += 1
        except Exception as exc:
            print(f"  ✗ {raw_path.name}: {exc}", file=sys.stderr)

    print()
    print(f"Re-warpeados: {rewarped}  |  Labels corregidos: {label_fixed}  |  Sin pool image: {skipped}")


if __name__ == "__main__":
    main()
