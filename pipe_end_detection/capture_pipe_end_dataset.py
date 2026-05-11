from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cam151_ref_detection.capture_history import (  # noqa: E402
    _camera_config_by_name,
    _capture_single_camera,
    _iso_now_local,
    _load_capture_config,
    _repo_rel,
    default_roi_paths,
)
from src.cam151_ref_detection.homography_preview import build_homography_preview  # noqa: E402
from src.cam151_ref_detection.roi_store import load_rois  # noqa: E402


CAPTURE_DATASET_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture camera pairs and export homography-warped images for pipe-end YOLO annotation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.json. Defaults to the shared workspace config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "pipe_end_detection" / "captures",
        help="Root folder for raw captures, warps, overlays, and manifests.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=REPO_ROOT / "pipe_end_detection" / "annotation_pool",
        help="Folder where warped images are copied for annotation.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. Defaults to YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--strict-ptz",
        action="store_true",
        help="Fail the run when a camera reports that it did not reach the requested PTZ target.",
    )
    parser.add_argument(
        "--keep-download-time-stamp",
        action="store_true",
        help="Keep timestamp overlays from config.json. Default disables them for cleaner training images.",
    )
    parser.add_argument(
        "--keep-camera-overlay",
        action="store_true",
        help="Keep camera overlays from config.json. Default disables them for cleaner training images.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep capturing on an interval until interrupted.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=60.0,
        help="Minutes between captures when --loop is enabled.",
    )
    return parser.parse_args()


def run_id_now() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def unique_run_id(output_root: Path, requested: str | None) -> str:
    base = (requested or run_id_now()).strip()
    if not base:
        raise ValueError("run_id cannot be empty.")
    candidate = base
    suffix = 1
    while (output_root / "runs" / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def resolve_local_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def clone_for_dataset_capture(
    config: dict[str, Any],
    *,
    keep_download_time_stamp: bool,
    keep_camera_overlay: bool,
) -> dict[str, Any]:
    cloned = json.loads(json.dumps(config))
    if not keep_download_time_stamp:
        cloned["stamp_download_time"] = False
    if not keep_camera_overlay:
        cloned["disable_overlay"] = True
    for camera_cfg in cloned.get("cameras") or []:
        if isinstance(camera_cfg, dict):
            if not keep_download_time_stamp:
                camera_cfg["stamp_download_time"] = False
            if not keep_camera_overlay:
                camera_cfg["disable_overlay"] = True
    return cloned


def is_cam152_side(side: str) -> bool:
    return str(side) == "152"


def build_warp_for_camera(
    *,
    side: str,
    image_path: Path,
    roi_path: Path,
    work_dir: Path,
    warp_dir: Path,
    overlay_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    roi_payload = load_rois(roi_path)
    flip = is_cam152_side(side)
    preview = build_homography_preview(
        image_path=image_path,
        lines=list(roi_payload.get("lines") or []),
        points=list(roi_payload.get("points") or []),
        output_dir=work_dir,
        src_points_override=roi_payload.get("src_points_override"),
        dst_rect_override=roi_payload.get("dst_rect_override"),
        flip_horizontal=flip,
        output_flip_horizontal=flip,
    )

    warp_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    warp_path = warp_dir / f"cam_{side}_{run_id}_warp.jpg"
    overlay_path = overlay_dir / f"cam_{side}_{run_id}_homography_overlay.jpg"
    shutil.copy2(preview.warp_path, warp_path)
    shutil.copy2(preview.overlay_path, overlay_path)

    return {
        "side": side,
        "roi_path": _repo_rel(roi_path),
        "warp_path": _repo_rel(warp_path),
        "overlay_path": _repo_rel(overlay_path),
        "source_image_path": _repo_rel(image_path),
        "output_size": {
            "width": int(preview.output_size[0]),
            "height": int(preview.output_size[1]),
        },
        "source_size": {
            "width": int(preview.source_size[0]),
            "height": int(preview.source_size[1]),
        },
        "src_points": [[float(x), float(y)] for x, y in preview.src_points],
        "dst_rect": [float(value) for value in preview.dst_rect],
        "mirror_rule": "cam152_backend_mirror" if flip else "none",
    }


def add_to_annotation_pool(
    *,
    side: str,
    warp_path: Path,
    annotation_root: Path,
    run_id: str,
) -> dict[str, str]:
    side_name = f"cam{side}"
    image_dir = annotation_root / "images" / side_name
    label_dir = annotation_root / "labels" / side_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    pooled_image = image_dir / f"{side_name}_{run_id}.jpg"
    pooled_label = label_dir / f"{side_name}_{run_id}.txt"
    shutil.copy2(warp_path, pooled_image)
    if not pooled_label.exists():
        pooled_label.write_text("", encoding="utf-8")
    return {
        "image_path": _repo_rel(pooled_image),
        "label_path": _repo_rel(pooled_label),
    }


def capture_once(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    annotation_root = args.annotation_root.resolve()
    run_id = unique_run_id(output_root, args.run_id)
    run_dir = output_root / "runs" / run_id
    raw_dir = run_dir / "raw"
    warp_dir = run_dir / "warped"
    overlay_dir = run_dir / "overlays"
    homography_work_dir = run_dir / "_homography_work"
    manifest_path = run_dir / "manifest.json"

    raw_dir.mkdir(parents=True, exist_ok=True)
    warp_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    homography_work_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "version": CAPTURE_DATASET_VERSION,
        "run_id": run_id,
        "captured_at": _iso_now_local(),
        "status": "starting",
        "purpose": "pipe_end_yolo_annotation",
        "run_dir": _repo_rel(run_dir),
        "raw_dir": _repo_rel(raw_dir),
        "warp_dir": _repo_rel(warp_dir),
        "overlay_dir": _repo_rel(overlay_dir),
        "annotation_root": _repo_rel(annotation_root),
        "config_path": None,
        "dataset_capture_overrides": {
            "stamp_download_time": bool(args.keep_download_time_stamp),
            "camera_overlay": bool(args.keep_camera_overlay),
        },
        "roi_paths": {},
        "cameras": {},
        "homography": {},
        "annotation_pool": {},
        "error": None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        loaded_config, config_path = _load_capture_config(args.config)
        capture_config = clone_for_dataset_capture(
            loaded_config,
            keep_download_time_stamp=bool(args.keep_download_time_stamp),
            keep_camera_overlay=bool(args.keep_camera_overlay),
        )
        manifest["config_path"] = _repo_rel(config_path)
        manifest["status"] = "capturing"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        user = str(capture_config.get("user") or "").strip()
        password = str(capture_config.get("password") or "").strip()
        if not user or not password:
            raise ValueError("Missing user/password in config.json.")

        for camera_name in ("cam151", "cam152"):
            camera_payload = _capture_single_camera(
                capture_config,
                _camera_config_by_name(capture_config, camera_name),
                user=user,
                password=password,
                run_id=run_id,
                raw_dir=raw_dir,
            )
            side = str(camera_payload["side"])
            manifest["cameras"][camera_name] = camera_payload
            if args.strict_ptz and not bool((camera_payload.get("ptz") or {}).get("reached_target", False)):
                raise RuntimeError(f"{camera_name} did not reach the requested PTZ target.")

        roi_paths = default_roi_paths()
        manifest["roi_paths"] = {f"cam{side}": _repo_rel(path) for side, path in roi_paths.items()}
        manifest["status"] = "warping"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        for camera_name, camera_payload in manifest["cameras"].items():
            side = str(camera_payload["side"])
            image_path = resolve_local_path(camera_payload["image_path"])
            roi_path = roi_paths[side]
            warp_info = build_warp_for_camera(
                side=side,
                image_path=image_path,
                roi_path=roi_path,
                work_dir=homography_work_dir / f"cam{side}",
                warp_dir=warp_dir,
                overlay_dir=overlay_dir,
                run_id=run_id,
            )
            manifest["homography"][camera_name] = warp_info
            pool_info = add_to_annotation_pool(
                side=side,
                warp_path=resolve_local_path(warp_info["warp_path"]),
                annotation_root=annotation_root,
                run_id=run_id,
            )
            manifest["annotation_pool"][camera_name] = pool_info

        manifest["status"] = "ready_for_annotation"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        raise


def loop(args: argparse.Namespace) -> None:
    interval_seconds = float(args.interval_minutes) * 60.0
    if interval_seconds <= 0:
        raise ValueError("--interval-minutes must be greater than zero.")
    while True:
        try:
            manifest = capture_once(args)
            print(f"Captured {manifest['run_id']}: {manifest['status']}")
        except Exception as exc:
            print(f"Capture failed: {exc}", file=sys.stderr)
        print(f"Sleeping {args.interval_minutes:g} minutes.")
        time.sleep(interval_seconds)


def main() -> None:
    args = parse_args()
    if args.loop:
        loop(args)
    else:
        manifest = capture_once(args)
        print("Pipe-end YOLO capture created.")
        print(f"- run_id: {manifest['run_id']}")
        print(f"- status: {manifest['status']}")
        print(f"- run_dir: {manifest['run_dir']}")
        for camera_name, pool_info in manifest["annotation_pool"].items():
            print(f"- {camera_name}: {pool_info['image_path']}")


if __name__ == "__main__":
    main()
