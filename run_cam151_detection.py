from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"  
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cam151_ref_detection import Cam151Config, run_cam151_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap processing for cam151 reference detection.")
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument("--out", default=None, help="Optional output directory")
    args = parser.parse_args()

    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(args.out) if args.out else ROOT / "artifacts" / image_path.stem
    summary = run_cam151_bootstrap(image_path, output_dir, Cam151Config())

    print(f"Processed: {image_path}")
    print(f"Artifacts: {output_dir}")
    print(f"Green components: {len(summary['green_components'])}")
    print(f"Bottom lines: {len(summary['bottom_lines'])}")


if __name__ == "__main__":
    main()
