"""Browser-based picker for warp scale reference lines."""

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


def _display_jupyter_link(url: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except Exception:
        return
    shell = get_ipython()
    if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
        return
    display(HTML(f'<a href="{url}" target="_blank">Abrir picker de lineas de escala</a>'))


def _clamp_point(value: Any, width: int, height: int) -> list[float] | None:
    if value is None:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite([x, y]).all():
        return None
    return [
        float(max(0.0, min(float(width), x))),
        float(max(0.0, min(float(height), y))),
    ]


def _normalize_existing_scales(
    scales: list[dict[str, Any]] | None,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, spec in enumerate(scales or [], start=1):
        if not isinstance(spec, dict):
            continue
        label = str(spec.get("label") or f"wscale_{index:02d}")
        distance_in = spec.get("distance_in")
        try:
            distance_value = float(distance_in)
        except (TypeError, ValueError):
            continue
        p1 = _clamp_point(spec.get("p1"), width, height)
        p2 = _clamp_point(spec.get("p2"), width, height)
        normalized.append(
            {
                "label": label,
                "distance_in": distance_value,
                "p1": p1,
                "p2": p2,
            }
        )
    return normalized


def _validate_result_lines(
    payload: Any,
    width: int,
    height: int,
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if not isinstance(payload, dict) or not payload.get("confirmed"):
        return None
    lines_payload = payload.get("lines")
    if not isinstance(lines_payload, dict):
        return None

    result: list[dict[str, Any]] = []
    for spec in specs:
        label = str(spec["label"])
        line_payload = lines_payload.get(label)
        if not isinstance(line_payload, dict):
            return None
        p1 = _clamp_point(line_payload.get("p1"), width, height)
        p2 = _clamp_point(line_payload.get("p2"), width, height)
        if p1 is None or p2 is None:
            return None
        result.append(
            {
                "label": label,
                "distance_in": float(spec["distance_in"]),
                "p1": p1,
                "p2": p2,
            }
        )
    return result


def _run_web_picker(
    image_path: str,
    scale_specs: list[dict[str, Any]],
    *,
    timeout: float | None = None,
) -> list[dict[str, Any]] | None:
    image_path_obj = Path(image_path)
    image_bytes = image_path_obj.read_bytes()
    image_mime = _image_mime_type(image_path)
    image_bgr = cv2.imread(str(image_path_obj), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    height, width = image_bgr.shape[:2]

    normalized_specs = _normalize_existing_scales(scale_specs, width, height)
    if not normalized_specs:
        raise ValueError("No hay lineas de escala configuradas para editar.")

    html_path = Path(__file__).with_name("scale_line_picker.html")
    html_template = html_path.read_text(encoding="utf-8")
    initial_state = {
        "lines": normalized_specs,
        "imageWidth": int(width),
        "imageHeight": int(height),
    }
    html = html_template.replace("__INITIAL_STATE__", json.dumps(initial_state, ensure_ascii=False))
    html_bytes = html.encode("utf-8")

    done = threading.Event()
    result_box: dict[str, list[dict[str, Any]] | None] = {"result": None}

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

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
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

            result_box["result"] = _validate_result_lines(payload, width, height, normalized_specs)
            done.set()
            self._send_json({"ok": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), PickerHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Picker web de lineas de escala listo: {url}")
    print("La celda queda esperando hasta que presiones Confirmar o Cancelar.")
    _display_jupyter_link(url)
    try:
        webbrowser.open(url, new=1)
    except Exception:
        pass

    try:
        if not done.wait(timeout=timeout):
            print("Picker web de lineas de escala: tiempo agotado sin resultado.")
            return None
        return result_box["result"]
    except KeyboardInterrupt:
        print("Picker web de lineas de escala cancelado desde el kernel.")
        return None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_scale_line_picker(
    image_bgr: np.ndarray | None,
    scale_specs: list[dict[str, Any]] | None = None,
    img_path: str | None = None,
    *,
    backend: str = "web",
    timeout: float | None = None,
) -> list[dict[str, Any]] | None:
    """Pick one or more warp-space scale lines in the displayed warped image."""
    backend_key = str(backend or "web").strip().lower()
    if backend_key not in {"web", "browser", "http"}:
        raise ValueError(f"Backend de picker de lineas de escala no soportado: {backend!r}")

    prepared_path, temp_path = _prepare_image_path(image_bgr, img_path)
    try:
        image = cv2.imread(prepared_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {prepared_path}")
        height, width = image.shape[:2]
        normalized_specs = _normalize_existing_scales(scale_specs, width, height)
        return _run_web_picker(prepared_path, normalized_specs, timeout=timeout)
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
