"""Browser-based picker for vertical reference line points."""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from .hom_picker_proc import _image_mime_type, _prepare_image_path


_REF_LABELS = ("mark_02", "mark_03")
_REF_NAMES = {
    "mark_02": "punto superior",
    "mark_03": "punto inferior",
}


def _display_jupyter_link(url: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except Exception:
        return
    shell = get_ipython()
    if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
        return
    display(HTML(f'<a href="{url}" target="_blank">Abrir picker de linea de referencia</a>'))


def _normalize_existing_points(
    existing_pts: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, list[float]]:
    normalized: dict[str, list[float]] = {}
    for label in _REF_LABELS:
        value = (existing_pts or {}).get(label)
        if value is None:
            continue
        try:
            x, y = float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not np.isfinite([x, y]).all():
            continue
        normalized[label] = [x, y]
    return normalized


def _validate_result_points(payload: Any, width: int, height: int) -> dict[str, list[float]] | None:
    if not isinstance(payload, dict) or not payload.get("confirmed"):
        return None
    points = payload.get("points")
    if not isinstance(points, dict):
        return None

    result: dict[str, list[float]] = {}
    for label in _REF_LABELS:
        value = points.get(label)
        if value is None:
            return None
        try:
            x, y = float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not np.isfinite([x, y]).all():
            return None
        result[label] = [x, y]
    return result


def _run_web_picker(
    image_path: str,
    existing_pts: dict[str, list[float]],
    *,
    timeout: float | None = None,
) -> dict[str, list[float]] | None:
    image_path_obj = Path(image_path)
    image_bytes = image_path_obj.read_bytes()
    image_mime = _image_mime_type(image_path)
    image_bgr = cv2.imread(str(image_path_obj), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    height, width = image_bgr.shape[:2]

    html_path = Path(__file__).with_name("ref_line_picker.html")
    html_template = html_path.read_text(encoding="utf-8")
    initial_state = {
        "existing": _normalize_existing_points(existing_pts, width, height),
        "labels": list(_REF_LABELS),
        "names": _REF_NAMES,
        "imageWidth": int(width),
        "imageHeight": int(height),
    }
    html = html_template.replace("__INITIAL_STATE__", json.dumps(initial_state, ensure_ascii=False))
    html_bytes = html.encode("utf-8")

    done = threading.Event()
    result_box: dict[str, dict[str, list[float]] | None] = {"result": None}

    class PickerHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_bytes(html_bytes, "text/html; charset=utf-8")
                return
            if parsed.path == "/image":
                self._send_bytes(image_bytes, image_mime)
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/result":
                self._send_json({"error": "not found"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw) if raw else {}
            except Exception as exc:
                self._send_json({"error": f"request invalido: {exc}"}, status=400)
                return

            result_box["result"] = _validate_result_points(payload, width, height)
            done.set()
            self._send_json({"ok": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), PickerHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Picker web de linea de referencia listo: {url}")
    print("La celda queda esperando hasta que presiones Confirmar o Cancelar.")
    _display_jupyter_link(url)
    try:
        webbrowser.open(url, new=1)
    except Exception:
        pass

    try:
        if not done.wait(timeout=timeout):
            print("Picker web de linea de referencia: tiempo agotado sin resultado.")
            return None
        return result_box["result"]
    except KeyboardInterrupt:
        print("Picker web de linea de referencia cancelado desde el kernel.")
        return None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_ref_line_picker(
    image_bgr: np.ndarray | None,
    existing_pts: dict[str, Any] | None = None,
    img_path: str | None = None,
    *,
    backend: str = "web",
    timeout: float | None = None,
) -> dict[str, list[float]] | None:
    """Pick mark_02 and mark_03 in displayed warp-image coordinates."""
    backend_key = str(backend or "web").strip().lower()
    if backend_key not in {"web", "browser", "http"}:
        raise ValueError(f"Backend de picker de linea no soportado: {backend!r}")

    prepared_path, temp_path = _prepare_image_path(image_bgr, img_path)
    try:
        image = cv2.imread(prepared_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {prepared_path}")
        height, width = image.shape[:2]
        normalized_existing = _normalize_existing_points(existing_pts, width, height)
        return _run_web_picker(prepared_path, normalized_existing, timeout=timeout)
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
