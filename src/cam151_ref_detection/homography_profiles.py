from __future__ import annotations

from datetime import datetime
import shutil
from pathlib import Path
from typing import Any

from .capture_history import default_roi_paths, repo_root
from .roi_store import load_rois, save_rois


LABELS = ("TL", "TR", "BL", "BR")


def normalize_side(side: Any) -> str:
    raw = str(side or "").strip().lower()
    if raw in {"151", "cam151", "cam_151", "left"}:
        return "151"
    if raw in {"152", "cam152", "cam_152", "right"}:
        return "152"
    raise ValueError(f"Camara no soportada: {side!r}")


def current_default_roi_path(side: Any) -> Path:
    side_key = normalize_side(side)
    return repo_root() / "manual_rois" / "_current_defaults" / f"cam_{side_key}_current_default_rois.toml"


def profile_root(side: Any) -> Path:
    side_key = normalize_side(side)
    return repo_root() / "manual_rois" / "homography_profiles" / f"cam{side_key}"


def profile_pointer_path(side: Any) -> Path:
    return profile_root(side) / "current_profile.txt"


def _profile_id_from_path(path: Path) -> str:
    return path.stem


def _profile_path_from_id(side: Any, profile_id: str) -> Path:
    raw = str(profile_id or "").strip()
    if not raw:
        raise ValueError("profile_id vacio.")
    if "/" in raw or "\\" in raw or raw in {".", ".."}:
        raise ValueError(f"profile_id invalido: {profile_id!r}")
    path = profile_root(side) / raw
    if path.suffix.lower() != ".toml":
        path = path.with_suffix(".toml")
    return path


def _normalize_points(points: Any) -> list[list[float]]:
    if isinstance(points, dict):
        normalized = []
        for label in LABELS:
            value = points.get(label)
            if value is None:
                raise ValueError(f"Falta punto {label}.")
            normalized.append([float(value[0]), float(value[1])])
        return normalized
    if isinstance(points, list) and len(points) == 4:
        return [[float(point[0]), float(point[1])] for point in points]
    raise ValueError("La homografia requiere 4 puntos.")


def list_profiles(side: Any) -> list[dict[str, Any]]:
    side_key = normalize_side(side)
    root = profile_root(side_key)
    pointer = ""
    pointer_path = profile_pointer_path(side_key)
    if pointer_path.exists():
        pointer = pointer_path.read_text(encoding="utf-8").strip()

    rows: list[dict[str, Any]] = []
    current_path = current_default_roi_path(side_key)
    if current_path.exists():
        payload = load_rois(current_path)
        rows.append(
            {
                "profile_id": "current_default",
                "label": f"Current default ({current_path.name})",
                "path": str(current_path),
                "updated_at": str(payload.get("updated_at") or ""),
                "image_name": str(payload.get("image_name") or ""),
                "is_current": True,
            }
        )

    if root.exists():
        for path in sorted(root.glob("*.toml"), key=lambda p: p.stat().st_mtime, reverse=True):
            profile_id = _profile_id_from_path(path)
            payload = load_rois(path)
            rows.append(
                {
                    "profile_id": profile_id,
                    "label": profile_id,
                    "path": str(path),
                    "updated_at": str(payload.get("updated_at") or ""),
                    "image_name": str(payload.get("image_name") or ""),
                    "is_current": profile_id == pointer,
                }
            )
    return rows


def resolve_profile_path(side: Any, profile_id: str | None = None) -> Path:
    side_key = normalize_side(side)
    raw = str(profile_id or "").strip()
    if not raw or raw == "current_default":
        return default_roi_paths()[side_key]
    path = _profile_path_from_id(side_key, raw)
    if not path.exists():
        raise FileNotFoundError(f"No existe perfil de homografia: {path}")
    return path


def set_default_profile(side: Any, profile_id: str) -> dict[str, Any]:
    side_key = normalize_side(side)
    source = resolve_profile_path(side_key, profile_id)
    target = current_default_roi_path(side_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        same_file = source.resolve() == target.resolve()
    except Exception:
        same_file = False
    if not same_file:
        shutil.copy2(source, target)
    root = profile_root(side_key)
    root.mkdir(parents=True, exist_ok=True)
    profile_pointer_path(side_key).write_text(
        "current_default" if str(profile_id).strip() == "current_default" else _profile_id_from_path(source),
        encoding="utf-8",
    )
    return {
        "camera": side_key,
        "profile_id": profile_id,
        "source_path": str(source),
        "default_path": str(target),
    }


def save_homography_profile(
    *,
    side: Any,
    image_path: str | Path,
    src_points: Any,
    base_profile_id: str | None = None,
    name: str | None = None,
    set_as_default: bool = True,
) -> dict[str, Any]:
    side_key = normalize_side(side)
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe imagen para guardar homografia: {image_path}")

    base_path = resolve_profile_path(side_key, base_profile_id)
    base_payload = load_rois(base_path)
    normalized_points = _normalize_points(src_points)

    root = profile_root(side_key)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name or "").strip()).strip("_")
    profile_id = f"cam{side_key}_homography_{stamp}" + (f"_{clean_name}" if clean_name else "")
    profile_path = root / f"{profile_id}.toml"

    save_rois(
        profile_path,
        image_path,
        list(base_payload.get("rois") or []),
        points=list(base_payload.get("points") or []),
        lines=list(base_payload.get("lines") or []),
        src_points_override=normalized_points,
        dst_rect_override=base_payload.get("dst_rect_override"),
    )

    default_result = None
    if set_as_default:
        default_result = set_default_profile(side_key, profile_id)
    return {
        "camera": side_key,
        "profile_id": profile_id,
        "profile_path": str(profile_path),
        "set_as_default": bool(set_as_default),
        "default_result": default_result,
    }
