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
    parser = argparse.ArgumentParser(description="Lanza el timeline de homografia y escribe la URL a un archivo.")
    parser.add_argument("--camera", type=str, default="152", choices=["151", "152"], help="Camara a revisar.")
    parser.add_argument("--port", type=int, default=None, help="Puerto fijo opcional.")
    parser.add_argument("--url-file", type=str, required=True, help="Archivo donde escribir la URL lista.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    url_path = Path(args.url_file)
    url_path.parent.mkdir(parents=True, exist_ok=True)
    url_path.write_text("PENDING", encoding="utf-8")
    handle = start_homography_timeline_server(
        camera=args.camera,
        port=args.port,
        open_browser=False,
        on_ready=lambda url: url_path.write_text(str(url), encoding="utf-8"),
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
