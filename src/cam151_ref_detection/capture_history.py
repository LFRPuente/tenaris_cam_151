from __future__ import annotations

import io
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from requests.auth import HTTPDigestAuth

from src.tenaris_tube_pipeline.config import build_camera_config, build_output_config
from src.tenaris_tube_pipeline.pair import process_tube_pair

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional dependency guard
    Image = None
    ImageDraw = None
    ImageFont = None


CAPTURE_HISTORY_VERSION = 1
VAPIX_SET_ZOOM = "/axis-cgi/com/ptz.cgi?zoom={zoom_level}"
VAPIX_SET_PAN_TILT = "/axis-cgi/com/ptz.cgi?pan={pan}&tilt={tilt}&speed={speed}"
VAPIX_SNAPSHOT = "/axis-cgi/jpg/image.cgi?resolution={resolution}&compression={compression}&camera={camera_id}"
DEFAULT_PTZ_MOVEMENT_SEQUENCE = "zoom_then_pan_tilt"
PRE_MOVE_ZOOM_PTZ_MOVEMENT_SEQUENCE = "pre_move_zoom_then_pan_tilt_then_zoom"
PTZ_STEP_DELAY_SECONDS = 0.5


class CaptureRunError(RuntimeError):
    def __init__(self, message: str, *, run_id: str | None = None, manifest_path: Path | None = None):
        super().__init__(message)
        self.run_id = run_id
        self.manifest_path = manifest_path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def capture_history_root() -> Path:
    return repo_root() / "artifacts" / "capture_history"


def latest_run_pointer_path() -> Path:
    return capture_history_root() / "latest_run.txt"


def default_roi_paths() -> dict[str, Path]:
    current_defaults = {
        "151": repo_root() / "manual_rois" / "_current_defaults" / "cam_151_current_default_rois.toml",
        "152": repo_root() / "manual_rois" / "_current_defaults" / "cam_152_current_default_rois.toml",
    }
    frozen_defaults = {
        "151": repo_root() / "manual_rois" / "_frozen_defaults" / "cam_151_202604022_rois.toml",
        "152": repo_root() / "manual_rois" / "_frozen_defaults" / "cam_152_202604022_rois.toml",
    }
    return {
        side: current_defaults[side] if current_defaults[side].exists() else frozen_defaults[side]
        for side in ("151", "152")
    }


def _repo_rel(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(repo_root().resolve())).replace("\\", "/")
    except Exception:
        return str(candidate)


def _safe_json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _iso_now_local() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _run_id_now() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _capture_run_dir(run_id: str) -> Path:
    return capture_history_root() / str(run_id).strip()


def _capture_manifest_path(run_id: str) -> Path:
    return _capture_run_dir(run_id) / "manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON invalido en {path}")
    return payload


def _write_manifest(manifest: dict[str, Any]) -> Path:
    run_id = str(manifest.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Manifest sin run_id.")
    path = _capture_manifest_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_capture_run_manifest(run_id: str) -> dict[str, Any]:
    path = _capture_manifest_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"No existe la corrida solicitada: {run_id}")
    return _read_json(path)


def load_latest_capture_run_manifest() -> dict[str, Any] | None:
    pointer_path = latest_run_pointer_path()
    if not pointer_path.exists():
        return None
    run_id = pointer_path.read_text(encoding="utf-8").strip()
    if not run_id:
        return None
    manifest_path = _capture_manifest_path(run_id)
    if not manifest_path.exists():
        return None
    return _read_json(manifest_path)


def delete_capture_run(run_id: str) -> dict[str, Any]:
    raw = str(run_id or "").strip()
    if not raw:
        raise ValueError("run_id vacio.")
    if "/" in raw or "\\" in raw or raw in {".", ".."}:
        raise ValueError(f"run_id invalido: {run_id!r}")
    run_dir = _capture_run_dir(raw)
    history_root = capture_history_root().resolve()
    target = run_dir.resolve()
    if history_root not in target.parents:
        raise ValueError(f"run_id fuera del historial: {run_id!r}")
    if not run_dir.exists():
        raise FileNotFoundError(f"No existe la corrida solicitada: {run_id}")

    pointer = latest_run_pointer_path()
    pointer_was_latest = False
    if pointer.exists():
        try:
            pointer_was_latest = pointer.read_text(encoding="utf-8").strip() == raw
        except Exception:
            pointer_was_latest = False

    shutil.rmtree(run_dir)
    if pointer_was_latest:
        try:
            pointer.unlink()
        except FileNotFoundError:
            pass

    return {
        "run_id": raw,
        "deleted_dir": _repo_rel(run_dir),
        "pointer_cleared": pointer_was_latest,
    }


def list_capture_run_manifests() -> list[dict[str, Any]]:
    root = capture_history_root()
    if not root.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for path in root.glob("*/manifest.json"):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        manifests.append(payload)
    manifests.sort(key=lambda item: str(item.get("captured_at") or item.get("run_id") or ""), reverse=True)
    return manifests


def _resolve_capture_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is not None:
        candidate = Path(config_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No existe config de captura: {config_path}")

    candidates = [
        repo_root() / "config.json",
        repo_root().parent / "config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No se encontro config.json para descarga de camaras.")


def _load_capture_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    resolved_path = _resolve_capture_config_path(config_path)
    payload = _read_json(resolved_path)
    return payload, resolved_path


def _camera_config_by_name(config_payload: dict[str, Any], camera_name: str) -> dict[str, Any]:
    wanted = str(camera_name).strip().lower()
    for camera_cfg in list(config_payload.get("cameras") or []):
        name = str(camera_cfg.get("name") or "").strip().lower()
        if name == wanted:
            return dict(camera_cfg)
    raise KeyError(f"No existe la camara {camera_name!r} en config.json")


def _http_get(base_url: str, endpoint: str, *, user: str, password: str, timeout: int = 20, headers: dict[str, str] | None = None) -> bytes:
    url = base_url.rstrip("/") + endpoint
    auth = HTTPDigestAuth(user, password)
    response = requests.get(url, auth=auth, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.content


def _http_post_form(
    base_url: str,
    endpoint: str,
    *,
    user: str,
    password: str,
    payload: dict[str, Any],
    timeout: int = 15,
) -> str:
    url = base_url.rstrip("/") + endpoint
    auth = HTTPDigestAuth(user, password)
    response = requests.post(url, auth=auth, data=payload, timeout=timeout)
    response.raise_for_status()
    return response.text.strip()


def _param_update(base_url: str, *, user: str, password: str, params: dict[str, Any]) -> None:
    text = _http_post_form(
        base_url,
        "/axis-cgi/param.cgi",
        user=user,
        password=password,
        payload={"action": "update", **params},
    )
    if text.startswith("# Error"):
        raise RuntimeError(text)


def _set_zoom(base_url: str, *, user: str, password: str, zoom_level: int) -> None:
    _http_get(
        base_url,
        VAPIX_SET_ZOOM.format(zoom_level=int(zoom_level)),
        user=user,
        password=password,
    )


def _set_pan_tilt(base_url: str, *, user: str, password: str, pan: float, tilt: float, speed: int) -> None:
    _http_get(
        base_url,
        VAPIX_SET_PAN_TILT.format(pan=float(pan), tilt=float(tilt), speed=int(speed)),
        user=user,
        password=password,
    )


def _get_ptz_position(base_url: str, *, user: str, password: str) -> dict[str, float]:
    text = _http_get(
        base_url,
        "/axis-cgi/com/ptz.cgi?query=position",
        user=user,
        password=password,
    ).decode("utf-8", errors="replace")
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip().lower()] = value.strip()
    return {
        "pan": float(values.get("pan", "0") or 0),
        "tilt": float(values.get("tilt", "0") or 0),
        "zoom": float(values.get("zoom", "0") or 0),
    }


def _wait_for_ptz_target(
    base_url: str,
    *,
    user: str,
    password: str,
    target: dict[str, float],
    speed: int,
    check_zoom: bool = True,
    check_pan: bool = True,
    check_tilt: bool = True,
) -> tuple[dict[str, float] | None, bool]:
    if speed >= 80:
        timeout_s = 6.0
    elif speed >= 30:
        timeout_s = 12.0
    else:
        timeout_s = 30.0
    deadline = time.time() + timeout_s
    last_pos: dict[str, float] | None = None
    stable_since: float | None = None
    while time.time() < deadline:
        try:
            current = _get_ptz_position(base_url, user=user, password=password)
        except Exception:
            time.sleep(0.2)
            continue
        if last_pos is not None:
            moved = (
                abs(current["zoom"] - last_pos["zoom"]) > 1.0
                or abs(current["pan"] - last_pos["pan"]) > 0.2
                or abs(current["tilt"] - last_pos["tilt"]) > 0.2
            )
            if moved:
                stable_since = None
            elif stable_since is None:
                stable_since = time.time()
        last_pos = current
        zoom_ok = (not check_zoom) or abs(current["zoom"] - float(target["zoom"])) <= 8.0
        pan_ok = (not check_pan) or abs(current["pan"] - float(target["pan"])) <= 0.8
        tilt_ok = (not check_tilt) or abs(current["tilt"] - float(target["tilt"])) <= 0.8
        if zoom_ok and pan_ok and tilt_ok:
            return current, True
        if stable_since is not None and (time.time() - stable_since) >= 0.8:
            return current, False
        time.sleep(0.15)
    return last_pos, False


def _capture_snapshot(
    base_url: str,
    *,
    user: str,
    password: str,
    resolution: str,
    compression: int,
    camera_id: int,
) -> bytes:
    endpoint = VAPIX_SNAPSHOT.format(
        resolution=resolution,
        compression=int(compression),
        camera_id=int(camera_id),
    )
    endpoint += f"&_ts={int(time.time() * 1000)}"
    return _http_get(
        base_url,
        endpoint,
        user=user,
        password=password,
        headers={
            "Cache-Control": "no-cache, no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )


def _text_bbox(draw: Any, text: str, font: Any) -> tuple[int, int, int, int]:
    if hasattr(draw, "textbbox"):
        return tuple(draw.textbbox((0, 0), text, font=font))
    width, height = draw.textsize(text, font=font)
    return (0, 0, int(width), int(height))


def _stamp_download_time(jpg_bytes: bytes, *, captured_at: datetime, label: str) -> bytes:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow es requerido para estampar la hora de descarga.")
    image = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"{label}: {captured_at.strftime('%Y-%m-%d %H:%M:%S')}"
    left, top, right, bottom = _text_bbox(draw, text, font)
    pad = 8
    x = 16
    y = 16
    draw.rectangle(
        (x - pad, y - pad, x + (right - left) + pad, y + (bottom - top) + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=(255, 255, 0), font=font)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()


def _merge_snapshot_config(global_cfg: dict[str, Any], camera_cfg: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(global_cfg.get("snapshot"), dict):
        merged.update(global_cfg["snapshot"])
    if isinstance(camera_cfg.get("snapshot"), dict):
        merged.update(camera_cfg["snapshot"])
    return merged


def _merge_ptz_config(global_cfg: dict[str, Any], camera_cfg: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(global_cfg.get("ptz"), dict):
        merged.update(global_cfg["ptz"])
    camera_ptz = camera_cfg.get("ptz")
    if camera_ptz is None:
        return merged
    if isinstance(camera_ptz, dict):
        merged.update(camera_ptz)
    return merged


def _merge_digital_framing_config(global_cfg: dict[str, Any], camera_cfg: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(global_cfg.get("digital_framing"), dict):
        merged.update(global_cfg["digital_framing"])
    camera_digital = camera_cfg.get("digital_framing")
    if camera_digital is None:
        return merged
    if isinstance(camera_digital, dict):
        merged.update(camera_digital)
    return merged


def _coerce_speed(raw_speed: Any, default: int = 100) -> int:
    if raw_speed is None:
        return int(default)
    return int(raw_speed)


def _resolve_resample_filter() -> Any:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _resolve_capture_ptz_for_digital_framing(
    digital_cfg: dict[str, Any],
    fallback_ptz_cfg: dict[str, Any],
) -> dict[str, Any]:
    capture_ptz: dict[str, Any] = {}
    if isinstance(fallback_ptz_cfg, dict):
        capture_ptz.update(fallback_ptz_cfg)
    override = digital_cfg.get("capture_ptz")
    if isinstance(override, dict):
        capture_ptz.update(override)
    if digital_cfg.get("force_wide_capture", True):
        capture_ptz["zoom"] = int(digital_cfg.get("capture_zoom", 0))
    return capture_ptz


def _apply_digital_framing_to_jpg(
    jpg_bytes: bytes,
    *,
    digital_cfg: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow es requerido para aplicar framing digital.")

    image = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    src_width, src_height = image.size
    zoom_factor = max(1.0, float(digital_cfg.get("zoom_factor") or 1.0))
    offset_x = float(digital_cfg.get("offset_x") or 0.0)
    offset_y = float(digital_cfg.get("offset_y") or 0.0)

    crop_width = max(1, int(round(src_width / zoom_factor)))
    crop_height = max(1, int(round(src_height / zoom_factor)))
    max_left = max(0, src_width - crop_width)
    max_top = max(0, src_height - crop_height)

    center_x = (src_width / 2.0) + (offset_x * crop_width)
    center_y = (src_height / 2.0) + (offset_y * crop_height)
    left = int(round(center_x - (crop_width / 2.0)))
    top = int(round(center_y - (crop_height / 2.0)))
    left = max(0, min(max_left, left))
    top = max(0, min(max_top, top))
    right = left + crop_width
    bottom = top + crop_height

    framed = image.crop((left, top, right, bottom))
    if framed.size != (src_width, src_height):
        framed = framed.resize((src_width, src_height), resample=_resolve_resample_filter())

    output = io.BytesIO()
    framed.save(output, format="JPEG", quality=95)
    return output.getvalue(), {
        "enabled": True,
        "zoom_factor": zoom_factor,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "crop_box": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        },
        "source_size": {
            "width": src_width,
            "height": src_height,
        },
        "output_size": {
            "width": src_width,
            "height": src_height,
        },
    }


def _resolve_ptz_movement_sequence(ptz_cfg: dict[str, Any]) -> str:
    raw = str(ptz_cfg.get("movement_sequence") or DEFAULT_PTZ_MOVEMENT_SEQUENCE).strip().lower()
    aliases = {
        "default": DEFAULT_PTZ_MOVEMENT_SEQUENCE,
        "legacy": DEFAULT_PTZ_MOVEMENT_SEQUENCE,
        "zoom_then_pan_tilt": DEFAULT_PTZ_MOVEMENT_SEQUENCE,
        "pre_move_zoom_then_pan_tilt_then_zoom": PRE_MOVE_ZOOM_PTZ_MOVEMENT_SEQUENCE,
        "prezoom_pan_tilt_zoom": PRE_MOVE_ZOOM_PTZ_MOVEMENT_SEQUENCE,
        "wide_pan_tilt_zoom": PRE_MOVE_ZOOM_PTZ_MOVEMENT_SEQUENCE,
        "zoom0_pan_tilt_zoom": PRE_MOVE_ZOOM_PTZ_MOVEMENT_SEQUENCE,
    }
    sequence = aliases.get(raw)
    if sequence is None:
        allowed = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"movement_sequence invalido: {raw!r}. Usa uno de: {allowed}")
    return sequence


def _apply_ptz_sequence(
    base_url: str,
    *,
    user: str,
    password: str,
    ptz_cfg: dict[str, Any],
) -> dict[str, Any]:
    applied = dict(ptz_cfg)
    zoom_value = ptz_cfg.get("zoom")
    pan_value = ptz_cfg.get("pan")
    tilt_value = ptz_cfg.get("tilt")
    speed_value = _coerce_speed(ptz_cfg.get("speed"), default=100)
    sequence = _resolve_ptz_movement_sequence(ptz_cfg)
    applied["movement_sequence"] = sequence

    final_zoom = int(zoom_value) if zoom_value is not None else None
    final_pan = float(pan_value) if pan_value is not None else None
    final_tilt = float(tilt_value) if tilt_value is not None else None
    actual_pos: dict[str, float] | None = None
    reached = True

    if sequence == PRE_MOVE_ZOOM_PTZ_MOVEMENT_SEQUENCE:
        pre_move_zoom_raw = ptz_cfg.get("pre_move_zoom", 0)
        pre_move_zoom = int(pre_move_zoom_raw) if pre_move_zoom_raw is not None else None
        applied["pre_move_zoom"] = pre_move_zoom
        if pre_move_zoom is not None and (final_pan is not None or final_tilt is not None):
            _set_zoom(base_url, user=user, password=password, zoom_level=pre_move_zoom)
            time.sleep(PTZ_STEP_DELAY_SECONDS)
        if final_pan is not None and final_tilt is not None:
            _set_pan_tilt(
                base_url,
                user=user,
                password=password,
                pan=final_pan,
                tilt=final_tilt,
                speed=speed_value,
            )
            actual_pos, reached = _wait_for_ptz_target(
                base_url,
                user=user,
                password=password,
                target={
                    "zoom": float(pre_move_zoom or 0),
                    "pan": final_pan,
                    "tilt": final_tilt,
                },
                speed=speed_value,
                check_zoom=False,
                check_pan=True,
                check_tilt=True,
            )
        elif final_zoom is None and pre_move_zoom is not None:
            actual_pos, reached = _wait_for_ptz_target(
                base_url,
                user=user,
                password=password,
                target={"zoom": float(pre_move_zoom), "pan": 0.0, "tilt": 0.0},
                speed=speed_value,
                check_zoom=True,
                check_pan=False,
                check_tilt=False,
            )
        if final_zoom is not None:
            _set_zoom(base_url, user=user, password=password, zoom_level=final_zoom)
            actual_pos, reached = _wait_for_ptz_target(
                base_url,
                user=user,
                password=password,
                target={
                    "zoom": float(final_zoom),
                    "pan": float(final_pan or 0.0),
                    "tilt": float(final_tilt or 0.0),
                },
                speed=speed_value,
                check_zoom=True,
                check_pan=final_pan is not None,
                check_tilt=final_tilt is not None,
            )
    else:
        if final_zoom is not None:
            _set_zoom(base_url, user=user, password=password, zoom_level=final_zoom)
            time.sleep(PTZ_STEP_DELAY_SECONDS)
        if final_pan is not None and final_tilt is not None:
            _set_pan_tilt(
                base_url,
                user=user,
                password=password,
                pan=final_pan,
                tilt=final_tilt,
                speed=speed_value,
            )
            actual_pos, reached = _wait_for_ptz_target(
                base_url,
                user=user,
                password=password,
                target={
                    "zoom": float(final_zoom or 0.0),
                    "pan": final_pan,
                    "tilt": final_tilt,
                },
                speed=speed_value,
                check_zoom=final_zoom is not None,
                check_pan=True,
                check_tilt=True,
            )
        elif final_zoom is not None:
            actual_pos, reached = _wait_for_ptz_target(
                base_url,
                user=user,
                password=password,
                target={"zoom": float(final_zoom), "pan": 0.0, "tilt": 0.0},
                speed=speed_value,
                check_zoom=True,
                check_pan=False,
                check_tilt=False,
            )

    applied["actual"] = actual_pos
    applied["reached_target"] = reached
    return applied


def _capture_single_camera(
    global_cfg: dict[str, Any],
    camera_cfg: dict[str, Any],
    *,
    user: str,
    password: str,
    run_id: str,
    raw_dir: Path,
) -> dict[str, Any]:
    camera_name = str(camera_cfg.get("name") or camera_cfg.get("ip") or "").strip()
    side = "151" if "151" in camera_name else "152" if "152" in camera_name else ""
    if side not in {"151", "152"}:
        raise ValueError(f"No se pudo inferir el lado de la camara: {camera_name}")
    ip = str(camera_cfg.get("ip") or "").strip()
    if not ip:
        raise ValueError(f"La camara {camera_name!r} no tiene IP configurada.")

    base_url = f"http://{ip}"
    snapshot_cfg = _merge_snapshot_config(global_cfg, camera_cfg)
    resolution = str(snapshot_cfg.get("resolution") or "2048x1536")
    compression = int(snapshot_cfg.get("compression") or 0)
    camera_id = int(snapshot_cfg.get("camera_id") or 1)

    ptz_cfg = _merge_ptz_config(global_cfg, camera_cfg)
    digital_framing_cfg = _merge_digital_framing_config(global_cfg, camera_cfg)
    digital_framing_enabled = bool(digital_framing_cfg.get("enabled", False))
    disable_overlay = bool(camera_cfg.get("disable_overlay", global_cfg.get("disable_overlay", False)))
    force_zoom_zero = bool(camera_cfg.get("force_zoom_zero", global_cfg.get("force_zoom_zero", False)))
    stamp_download_time_enabled = bool(camera_cfg.get("stamp_download_time", global_cfg.get("stamp_download_time", False)))
    stamp_download_time_label = str(camera_cfg.get("download_time_label", global_cfg.get("download_time_label", "Descarga")))

    if disable_overlay:
        _param_update(
            base_url,
            user=user,
            password=password,
            params={"Image.I0.Appearance.Overlays": "off"},
        )

    capture_ptz_cfg = ptz_cfg
    if digital_framing_enabled:
        capture_ptz_cfg = _resolve_capture_ptz_for_digital_framing(digital_framing_cfg, ptz_cfg)

    if capture_ptz_cfg:
        capture_ptz_cfg = _apply_ptz_sequence(
            base_url,
            user=user,
            password=password,
            ptz_cfg=capture_ptz_cfg,
        )
    elif force_zoom_zero:
        _set_zoom(base_url, user=user, password=password, zoom_level=0)
        time.sleep(PTZ_STEP_DELAY_SECONDS)

    jpg_bytes = _capture_snapshot(
        base_url,
        user=user,
        password=password,
        resolution=resolution,
        compression=compression,
        camera_id=camera_id,
    )
    captured_at = datetime.now().astimezone()
    if stamp_download_time_enabled:
        jpg_bytes = _stamp_download_time(jpg_bytes, captured_at=captured_at, label=stamp_download_time_label)

    wide_image_name: str | None = None
    wide_image_path: str | None = None
    digital_framing_info: dict[str, Any] | None = None
    if digital_framing_enabled:
        wide_image_name = f"cam_{side}_{run_id}__wide_source.jpg"
        wide_image_file = raw_dir / wide_image_name
        wide_image_file.write_bytes(jpg_bytes)
        wide_image_path = _repo_rel(wide_image_file)
        jpg_bytes, digital_framing_info = _apply_digital_framing_to_jpg(
            jpg_bytes,
            digital_cfg=digital_framing_cfg,
        )

    filename = f"cam_{side}_{run_id}.jpg"
    image_path = raw_dir / filename
    image_path.write_bytes(jpg_bytes)

    payload = {
        "side": side,
        "camera_name": camera_name,
        "ip": ip,
        "image_name": filename,
        "image_path": _repo_rel(image_path),
        "captured_at": captured_at.replace(microsecond=0).isoformat(),
        "snapshot": {
            "resolution": resolution,
            "compression": compression,
            "camera_id": camera_id,
        },
        "ptz": _safe_json_clone(capture_ptz_cfg),
        "disable_overlay": disable_overlay,
        "stamp_download_time": stamp_download_time_enabled,
    }
    if wide_image_name is not None and wide_image_path is not None:
        payload["wide_source_image_name"] = wide_image_name
        payload["wide_source_image_path"] = wide_image_path
    if digital_framing_info is not None:
        payload["digital_framing"] = {
            **digital_framing_info,
            "capture_ptz": _safe_json_clone(capture_ptz_cfg),
        }
    return payload


def _copy_promoted_file(source_path: Path, destination_path: Path) -> str:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return _repo_rel(destination_path)


def _promote_run_to_latest(manifest: dict[str, Any]) -> dict[str, Any]:
    processing = dict(manifest.get("processing") or {})
    matcher_input_dir = repo_root() / "artifacts" / "tube_matcher_inputs"
    match_output_dir = repo_root() / "artifacts" / "tube_matching"

    latest_map = {
        "cam151_dataset_latest_path": (
            repo_root() / str(processing.get("cam151_dataset_latest_path") or ""),
            matcher_input_dir / "cam151_tube_measurements_latest.json",
        ),
        "cam152_dataset_latest_path": (
            repo_root() / str(processing.get("cam152_dataset_latest_path") or ""),
            matcher_input_dir / "cam152_tube_measurements_latest.json",
        ),
        "match_latest_json_path": (
            repo_root() / str(processing.get("match_latest_json_path") or ""),
            match_output_dir / "tube_match_latest.json",
        ),
        "match_latest_xlsx_path": (
            repo_root() / str(processing.get("match_latest_xlsx_path") or ""),
            match_output_dir / "tube_match_latest.xlsx",
        ),
    }

    promoted: dict[str, Any] = {
        "synced_at": _iso_now_local(),
        "targets": {},
    }
    for label, (source_path, destination_path) in latest_map.items():
        if not source_path.exists():
            raise FileNotFoundError(f"No existe archivo a promover: {source_path}")
        promoted["targets"][label] = _copy_promoted_file(source_path, destination_path)

    pointer_path = latest_run_pointer_path()
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(str(manifest.get("run_id") or ""), encoding="utf-8")
    promoted["latest_run_pointer"] = _repo_rel(pointer_path)
    return promoted


def _pipe_end_yolo_manifest_entry(result: Any) -> dict[str, Any]:
    yolo_result = getattr(result, "pipe_end_yolo", None)
    if yolo_result is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "model_path": _repo_rel(yolo_result.model_path),
        "prediction_count": int(yolo_result.count),
        "predictions_path": _repo_rel(yolo_result.predictions_path),
        "overlay_path": _repo_rel(yolo_result.overlay_path),
        "image_path": _repo_rel(yolo_result.image_path),
        "image_size": {
            "width": int(yolo_result.image_width),
            "height": int(yolo_result.image_height),
        },
        "imgsz": int(yolo_result.imgsz),
        "conf": float(yolo_result.conf),
        "iou": float(yolo_result.iou),
        "device": yolo_result.device,
    }


def capture_and_process_pair(*, config_path: str | Path | None = None) -> dict[str, Any]:
    run_id = _run_id_now()
    run_dir = _capture_run_dir(run_id)
    if run_dir.exists():
        suffix = 1
        while True:
            candidate_id = f"{run_id}_{suffix:02d}"
            candidate_dir = _capture_run_dir(candidate_id)
            if not candidate_dir.exists():
                run_id = candidate_id
                run_dir = candidate_dir
                break
            suffix += 1

    raw_dir = run_dir / "raw"
    processing_root = run_dir / "processing"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processing_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "version": CAPTURE_HISTORY_VERSION,
        "run_id": run_id,
        "captured_at": _iso_now_local(),
        "status": "starting",
        "source": "camera_capture",
        "run_dir": _repo_rel(run_dir),
        "raw_dir": _repo_rel(raw_dir),
        "processing_root": _repo_rel(processing_root),
        "config_path": None,
        "roi_paths": {},
        "cameras": {},
        "processing": {},
        "latest_sync": {},
        "error": None,
    }
    manifest_path = _write_manifest(manifest)

    try:
        global_cfg, resolved_config_path = _load_capture_config(config_path)
        manifest["config_path"] = _repo_rel(resolved_config_path)
        manifest["status"] = "capturing"
        _write_manifest(manifest)

        user = str(global_cfg.get("user") or "").strip()
        password = str(global_cfg.get("password") or "").strip()
        if not user or not password:
            raise ValueError("Faltan credenciales user/password en config.json.")

        cam151_capture = _capture_single_camera(
            global_cfg,
            _camera_config_by_name(global_cfg, "cam151"),
            user=user,
            password=password,
            run_id=run_id,
            raw_dir=raw_dir,
        )
        cam152_capture = _capture_single_camera(
            global_cfg,
            _camera_config_by_name(global_cfg, "cam152"),
            user=user,
            password=password,
            run_id=run_id,
            raw_dir=raw_dir,
        )
        manifest["cameras"] = {
            "cam151": cam151_capture,
            "cam152": cam152_capture,
        }
        manifest["status"] = "captured"
        _write_manifest(manifest)

        roi_paths = default_roi_paths()
        for side_key, roi_path in roi_paths.items():
            if not roi_path.exists():
                raise FileNotFoundError(f"No existe ROI configurado para cam{side_key}: {roi_path}")
        manifest["roi_paths"] = {
            "cam151": _repo_rel(roi_paths["151"]),
            "cam152": _repo_rel(roi_paths["152"]),
        }

        outputs = build_output_config(
            artifact_root=processing_root / "backend_tube_pipeline",
            matcher_input_dir=processing_root / "tube_matcher_inputs",
            match_output_dir=processing_root / "tube_matching",
        )
        cam151_cfg = build_camera_config(
            "151",
            repo_root() / str(cam151_capture["image_path"]),
            roi_paths["151"],
            dataset_name="cam151",
            source_name="backend_tube_pipeline_cam151_capture",
        )
        cam152_cfg = build_camera_config(
            "152",
            repo_root() / str(cam152_capture["image_path"]),
            roi_paths["152"],
            dataset_name="cam152",
            source_name="backend_tube_pipeline_cam152_capture",
        )

        manifest["status"] = "processing"
        _write_manifest(manifest)
        result = process_tube_pair(
            cam151_cfg,
            cam152_cfg,
            outputs,
            output_stem=f"capture_pair_{run_id}",
        )
        summary = dict(result.match_payload.get("summary") or {})
        pipe_end_yolo = {
            "cam151": _pipe_end_yolo_manifest_entry(result.cam151),
            "cam152": _pipe_end_yolo_manifest_entry(result.cam152),
        }
        manifest["processing"] = {
            "status": "processed",
            "artifact_root": _repo_rel(outputs.artifact_root),
            "matcher_input_dir": _repo_rel(outputs.matcher_input_dir),
            "match_output_dir": _repo_rel(outputs.match_output_dir),
            "cam151_historical_dataset_path": _repo_rel(result.cam151.measurement_export_path),
            "cam152_historical_dataset_path": _repo_rel(result.cam152.measurement_export_path),
            "cam151_dataset_latest_path": _repo_rel(outputs.matcher_input_dir / "cam151_tube_measurements_latest.json"),
            "cam152_dataset_latest_path": _repo_rel(outputs.matcher_input_dir / "cam152_tube_measurements_latest.json"),
            "match_json_path": _repo_rel(result.result_json_path),
            "match_xlsx_path": _repo_rel(result.result_xlsx_path),
            "match_latest_json_path": _repo_rel(result.latest_json_path),
            "match_latest_xlsx_path": _repo_rel(result.latest_xlsx_path),
            "summary": {
                "matched": int(summary.get("matched") or 0),
                "left_only": int(summary.get("left_only") or 0),
                "right_only": int(summary.get("right_only") or 0),
            },
            "tube_counts": {
                "cam151": int(result.cam151.tube_count),
                "cam152": int(result.cam152.tube_count),
            },
            "detection_source": "yolo_pipe_end" if pipe_end_yolo["cam151"].get("enabled") or pipe_end_yolo["cam152"].get("enabled") else "notebook_style_sobel_x_roi",
            "pipe_end_yolo": pipe_end_yolo,
        }
        manifest["latest_sync"] = _promote_run_to_latest(manifest)
        manifest["status"] = "processed"
        _write_manifest(manifest)
        return manifest
    except Exception as exc:
        processing = dict(manifest.get("processing") or {})
        if processing.get("status") == "processed":
            manifest["status"] = "processed_with_warning"
        elif manifest.get("cameras"):
            manifest["status"] = "processing_failed"
        else:
            manifest["status"] = "capture_failed"
        manifest["error"] = str(exc)
        manifest_path = _write_manifest(manifest)
        raise CaptureRunError(str(exc), run_id=run_id, manifest_path=manifest_path) from exc
