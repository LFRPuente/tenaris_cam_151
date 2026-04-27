from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraPipelineConfig:
    side: str
    image_path: Path
    roi_path: Path
    dataset_name: str
    source_name: str


@dataclass(frozen=True)
class PipelineOutputConfig:
    artifact_root: Path
    matcher_input_dir: Path
    match_output_dir: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_artifact_root() -> Path:
    return repo_root() / "artifacts" / "backend_tube_pipeline"


def default_matcher_input_dir() -> Path:
    return repo_root() / "artifacts" / "tube_matcher_inputs"


def default_match_output_dir() -> Path:
    return repo_root() / "artifacts" / "tube_matching"


def resolve_existing_path(raw_path: str | Path, *, search_dirs: list[Path] | None = None) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    base_dirs = search_dirs or []
    for base_dir in base_dirs:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
        candidate = base_dir / path.name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No existe la ruta requerida: {raw_path}")


def default_roi_for_image(image_path: str | Path) -> Path:
    image = Path(image_path)
    candidate = repo_root() / "manual_rois" / f"{image.stem}_rois.toml"
    if not candidate.exists():
        raise FileNotFoundError(f"No se encontro ROI default para {image.name}: {candidate}")
    return candidate


def build_camera_config(
    side: str,
    image_path: str | Path,
    roi_path: str | Path | None = None,
    *,
    dataset_name: str | None = None,
    source_name: str | None = None,
) -> CameraPipelineConfig:
    side_key = str(side).strip()
    if side_key not in {"151", "152"}:
        raise ValueError(f"Camara no soportada: {side!r}")

    root = repo_root()
    image = resolve_existing_path(image_path, search_dirs=[root, root / "test_images"])
    roi = resolve_existing_path(roi_path, search_dirs=[root, root / "manual_rois"]) if roi_path else default_roi_for_image(image)

    return CameraPipelineConfig(
        side=side_key,
        image_path=image,
        roi_path=roi,
        dataset_name=dataset_name or f"cam{side_key}",
        source_name=source_name or f"backend_tube_pipeline_cam{side_key}",
    )


def build_output_config(
    *,
    artifact_root: str | Path | None = None,
    matcher_input_dir: str | Path | None = None,
    match_output_dir: str | Path | None = None,
) -> PipelineOutputConfig:
    return PipelineOutputConfig(
        artifact_root=Path(artifact_root) if artifact_root else default_artifact_root(),
        matcher_input_dir=Path(matcher_input_dir) if matcher_input_dir else default_matcher_input_dir(),
        match_output_dir=Path(match_output_dir) if match_output_dir else default_match_output_dir(),
    )
