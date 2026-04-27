from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cam151_ref_detection.sorting_table_mvp_proc import start_sorting_table_mvp_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lanza la app cliente Sorting Table MVP.")
    parser.add_argument("--match", type=str, default=None, help="Ruta al resultado JSON de tube matching.")
    parser.add_argument("--cam151", type=str, default=None, help="Ruta al dataset cam151 si quieres fijarlo.")
    parser.add_argument("--cam152", type=str, default=None, help="Ruta al dataset cam152 si quieres fijarlo.")
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automaticamente.")
    parser.add_argument("--port", type=int, default=None, help="Puerto fijo opcional.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    handle = start_sorting_table_mvp_server(
        match_source=args.match,
        cam151_input=args.cam151,
        cam152_input=args.cam152,
        open_browser=not args.no_browser,
        port=args.port,
    )
    stop_event = threading.Event()
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
