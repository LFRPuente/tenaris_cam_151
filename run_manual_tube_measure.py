from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cam151_ref_detection.manual_tube_measure_proc import run_manual_tube_measure_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lanza la herramienta manual de medicion de tubos.")
    parser.add_argument("--cam151", type=str, default=None, help="Ruta al JSON exportado de cam151.")
    parser.add_argument("--cam152", type=str, default=None, help="Ruta al JSON exportado de cam152.")
    parser.add_argument("--input-dir", type=str, default=None, help="Directorio donde buscar los exports latest.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directorio para escribir el resultado final.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_manual_tube_measure_app(
        args.cam151,
        args.cam152,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        open_browser=True,
    )
    if result.confirmed:
        print(f"Resultado JSON: {result.result_json_path}")
        print(f"Resultado XLSX: {result.result_xlsx_path}")
        return 0
    print("Medicion manual cancelada o sin confirmacion.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
