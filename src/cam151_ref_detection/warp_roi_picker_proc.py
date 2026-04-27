"""Browser-based ROI picker for already warped images."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class FullWarpView:
    image_bgr: np.ndarray
    warp_path: Path
    view_rect: tuple[float, float, float, float]
    output_mirrored: bool


def _display_jupyter_link(url: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except Exception:
        return
    shell = get_ipython()
    if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
        return
    display(HTML(f'<a href="{url}" target="_blank">Abrir picker de ROI del warp</a>'))


def build_full_warp_view(
    image_path: str | Path,
    homography_matrix: list[list[float]] | np.ndarray,
    output_dir: str | Path,
    *,
    flip_horizontal: bool = False,
    output_flip_horizontal: bool = False,
) -> FullWarpView:
    """
    Warp the full source image into the homography plane.

    build_homography_preview(..., dst_rect_override=None) intentionally returns
    only the TL/TR/BL/BR base rectangle when src_points_override is exact. This
    helper instead projects the complete source-image bounds and includes every
    visible transformed pixel, including negative coordinates in homography
    space.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    if flip_horizontal:
        image_bgr = cv2.flip(image_bgr, 1)

    height, width = image_bgr.shape[:2]
    transform = np.asarray(homography_matrix, dtype=np.float64)
    corners = np.asarray(
        [[[0.0, 0.0]], [[float(width - 1), 0.0]], [[float(width - 1), float(height - 1)]], [[0.0, float(height - 1)]]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(corners, transform.astype(np.float32)).reshape(-1, 2)
    if projected.size == 0 or not np.all(np.isfinite(projected)):
        raise ValueError("No se pudo proyectar la imagen completa al plano de homografia.")

    min_x = float(np.floor(np.min(projected[:, 0])))
    min_y = float(np.floor(np.min(projected[:, 1])))
    max_x = float(np.ceil(np.max(projected[:, 0])))
    max_y = float(np.ceil(np.max(projected[:, 1])))
    out_w = max(80, int(round(max_x - min_x)))
    out_h = max(80, int(round(max_y - min_y)))

    translation = np.asarray(
        [[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    view_transform = translation @ transform
    warp = cv2.warpPerspective(image_bgr, view_transform, (out_w, out_h))
    if output_flip_horizontal:
        warp = cv2.flip(warp, 1)

    warp_path = output_dir / "full_source_warp.jpg"
    cv2.imwrite(str(warp_path), warp)
    return FullWarpView(
        image_bgr=warp,
        warp_path=warp_path,
        view_rect=(min_x, min_y, min_x + float(out_w), min_y + float(out_h)),
        output_mirrored=bool(output_flip_horizontal),
    )


def _normalize_rect(rect: object, width: int, height: int) -> list[float] | None:
    if not rect:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in rect]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    x0, x1 = sorted((max(0.0, min(float(width), x0)), max(0.0, min(float(width), x1))))
    y0, y1 = sorted((max(0.0, min(float(height), y0)), max(0.0, min(float(height), y1))))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    return [x0, y0, x1, y1]


def _validate_result_rect(payload: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(payload, dict) or not payload.get("confirmed"):
        return None
    return _normalize_rect(payload.get("rect"), width, height)


def _run_web_picker(
    image_path: str,
    existing_rect: list[float] | None,
    *,
    timeout: float | None = None,
) -> list[float] | None:
    image_path_obj = Path(image_path)
    image_bytes = image_path_obj.read_bytes()
    image_mime = _image_mime_type(image_path)
    image_bgr = cv2.imread(str(image_path_obj), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    height, width = image_bgr.shape[:2]

    html_path = Path(__file__).with_name("warp_roi_picker.html")
    html_template = html_path.read_text(encoding="utf-8")
    initial_state = {
        "existing": _normalize_rect(existing_rect, width, height),
        "imageWidth": int(width),
        "imageHeight": int(height),
    }
    html = html_template.replace("__INITIAL_STATE__", json.dumps(initial_state, ensure_ascii=False))
    html_bytes = html.encode("utf-8")

    done = threading.Event()
    result_box: dict[str, list[float] | None] = {"result": None}

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

            result_box["result"] = _validate_result_rect(payload, width, height)
            done.set()
            self._send_json({"ok": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), PickerHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Picker web de ROI listo: {url}")
    print("La celda queda esperando hasta que presiones Confirmar o Cancelar.")
    _display_jupyter_link(url)
    try:
        webbrowser.open(url, new=1)
    except Exception:
        pass

    try:
        if not done.wait(timeout=timeout):
            print("Picker web de ROI: tiempo agotado sin resultado.")
            return None
        return result_box["result"]
    except KeyboardInterrupt:
        print("Picker web de ROI cancelado desde el kernel.")
        return None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_warp_roi_picker(
    image_bgr: np.ndarray | None,
    existing_rect: list[float] | tuple[float, float, float, float] | None = None,
    img_path: str | None = None,
    *,
    backend: str = "web",
    timeout: float | None = None,
) -> list[float] | None:
    """Pick [x0, y0, x1, y1] in the displayed warped image coordinates."""
    backend_key = str(backend or "web").strip().lower()
    if backend_key not in {"web", "browser", "http"}:
        raise ValueError(f"Backend de picker ROI no soportado: {backend!r}")

    prepared_path, temp_path = _prepare_image_path(image_bgr, img_path)
    try:
        image = cv2.imread(prepared_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {prepared_path}")
        height, width = image.shape[:2]
        normalized_existing = _normalize_rect(existing_rect, width, height)
        return _run_web_picker(prepared_path, normalized_existing, timeout=timeout)
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
