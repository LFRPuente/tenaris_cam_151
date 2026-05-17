from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cam151_ref_detection.homography_timeline_proc import start_homography_timeline_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lanza la app de timeline de homografia.")
    parser.add_argument("--camera", type=str, default="152", choices=["151", "152"], help="Camara a revisar.")
    parser.add_argument("--port", type=int, default=None, help="Puerto fijo opcional.")
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automaticamente.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    handle = start_homography_timeline_server(
        camera=args.camera,
        port=args.port,
        open_browser=not args.no_browser,
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
