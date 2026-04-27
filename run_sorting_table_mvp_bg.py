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
    parser = argparse.ArgumentParser(description="Lanza Sorting Table MVP y escribe la URL a un archivo.")
    parser.add_argument("--match", type=str, default=None, help="Ruta al resultado JSON de tube matching.")
    parser.add_argument("--cam151", type=str, default=None, help="Ruta al dataset cam151 si quieres fijarlo.")
    parser.add_argument("--cam152", type=str, default=None, help="Ruta al dataset cam152 si quieres fijarlo.")
    parser.add_argument("--port", type=int, default=None, help="Puerto fijo opcional.")
    parser.add_argument("--url-file", type=str, required=True, help="Archivo donde escribir la URL cuando quede lista.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    url_path = Path(args.url_file)
    url_path.parent.mkdir(parents=True, exist_ok=True)
    url_path.write_text("PENDING", encoding="utf-8")

    handle = start_sorting_table_mvp_server(
        match_source=args.match,
        cam151_input=args.cam151,
        cam152_input=args.cam152,
        open_browser=False,
        port=args.port,
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
