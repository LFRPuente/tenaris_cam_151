from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tenaris_tube_pipeline.config import build_camera_config, build_output_config
from src.tenaris_tube_pipeline.pair import process_tube_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Procesa un par cam151/cam152 y alimenta la Sorting Table MVP.")
    parser.add_argument("--cam151-image", required=True, help="Imagen RAW de cam151.")
    parser.add_argument("--cam152-image", required=True, help="Imagen RAW de cam152.")
    parser.add_argument("--cam151-roi", default=None, help="ROI TOML de cam151. Default: manual_rois/<stem>_rois.toml")
    parser.add_argument("--cam152-roi", default=None, help="ROI TOML de cam152. Default: manual_rois/<stem>_rois.toml")
    parser.add_argument("--artifact-root", default=None, help="Directorio para artefactos internos del pipeline.")
    parser.add_argument("--matcher-input-dir", default=None, help="Directorio de exports cam151/cam152 para el matcher.")
    parser.add_argument("--match-output-dir", default=None, help="Directorio de resultados tube_match_latest.")
    parser.add_argument("--output-stem", default=None, help="Stem opcional para el resultado historico del match.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = build_output_config(
        artifact_root=args.artifact_root,
        matcher_input_dir=args.matcher_input_dir,
        match_output_dir=args.match_output_dir,
    )
    cam151 = build_camera_config(
        "151",
        args.cam151_image,
        args.cam151_roi,
        dataset_name="cam151",
        source_name="backend_tube_pipeline_cam151",
    )
    cam152 = build_camera_config(
        "152",
        args.cam152_image,
        args.cam152_roi,
        dataset_name="cam152",
        source_name="backend_tube_pipeline_cam152",
    )
    result = process_tube_pair(cam151, cam152, outputs, output_stem=args.output_stem)
    summary = result.match_payload.get("summary") or {}
    print(f"cam151 tubos: {result.cam151.tube_count}")
    print(f"cam152 tubos: {result.cam152.tube_count}")
    print(f"matches: {summary.get('matched')} | left_only: {summary.get('left_only')} | right_only: {summary.get('right_only')}")
    print(f"resultado JSON: {result.result_json_path}")
    print(f"resultado XLSX: {result.result_xlsx_path}")
    print(f"latest JSON: {result.latest_json_path}")
    print(f"latest XLSX: {result.latest_xlsx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
