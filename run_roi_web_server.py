from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from src.cam151_ref_detection.homography_preview import build_homography_preview
from src.cam151_ref_detection.roi_store import load_rois, save_rois
from src.cam151_ref_detection.tube_detection_preview import build_tube_detection_preview


ROOT = Path(__file__).resolve().parent
TEST_IMAGES_DIR = ROOT / "test_images"
MANUAL_ROIS_DIR = ROOT / "manual_rois"
PREVIEW_DIR = ROOT / "artifacts" / "web_homography_preview"


class RoiWebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _json_response(self, payload: dict | list, status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/images":
            images = [
                {
                    "name": path.name,
                    "url": f"/test_images/{path.name}",
                }
                for path in sorted(TEST_IMAGES_DIR.glob("cam151_wide_*.jpg"))
            ]
            self._json_response({"images": images})
            return

        if parsed.path == "/api/rois":
            query = parse_qs(parsed.query)
            image_name = unquote(query.get("image", [""])[0]).strip()
            if not image_name:
                self._json_response({"error": "Missing image query parameter."}, status=400)
                return
            payload = load_rois(MANUAL_ROIS_DIR / f"{Path(image_name).stem}_rois.toml")
            self._json_response(payload)
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/homography_preview":
            self._handle_homography_preview()
            return
        if parsed.path == "/api/tube_detection_preview":
            self._handle_tube_detection_preview()
            return

        if parsed.path != "/api/rois":
            self._json_response({"error": "Unknown endpoint."}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image_name = str(payload["image_name"])
            rois = list(payload.get("rois", []))
            points = list(payload.get("points", []))
            lines = list(payload.get("lines", []))
        except Exception as exc:
            self._json_response({"error": f"Invalid request: {exc}"}, status=400)
            return

        image_path = TEST_IMAGES_DIR / image_name
        if not image_path.exists():
            self._json_response({"error": f"Image not found: {image_name}"}, status=404)
            return

        MANUAL_ROIS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = MANUAL_ROIS_DIR / f"{image_path.stem}_rois.toml"
        saved = save_rois(save_path, image_path, rois, points=points, lines=lines)
        self._json_response(saved)

    def _handle_homography_preview(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image_name = str(payload["image_name"])
            points = list(payload.get("points", []))
            lines = list(payload.get("lines", []))
            warp_padding = dict(payload.get("warp_padding", {}))
            src_points_override = list(payload.get("src_points_override", []))
            dst_rect_override = payload.get("dst_rect_override")
        except Exception as exc:
            self._json_response({"error": f"Invalid request: {exc}"}, status=400)
            return

        image_path = TEST_IMAGES_DIR / image_name
        if not image_path.exists():
            self._json_response({"error": f"Image not found: {image_name}"}, status=404)
            return

        preview_dir = PREVIEW_DIR / image_path.stem
        try:
            result = build_homography_preview(
                image_path=image_path,
                lines=lines,
                points=points,
                output_dir=preview_dir,
                warp_padding=warp_padding,
                src_points_override=src_points_override,
                dst_rect_override=dst_rect_override,
            )
        except Exception as exc:
            self._json_response({"error": str(exc)}, status=400)
            return

        self._json_response(
            {
                "overlay_url": f"/artifacts/web_homography_preview/{image_path.stem}/{result.overlay_path.name}",
                "warp_url": f"/artifacts/web_homography_preview/{image_path.stem}/{result.warp_path.name}",
                "used_line_labels": result.used_line_labels,
                "src_points": result.src_points,
                "output_size": result.output_size,
                "base_size": result.base_size,
                "warp_padding": result.warp_padding,
                "dst_rect": result.dst_rect,
                "homography_matrix": result.homography_matrix,
                "inverse_homography_matrix": result.inverse_homography_matrix,
            }
        )

    def _handle_tube_detection_preview(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image_name = str(payload["image_name"])
            rois = list(payload.get("rois", []))
            points = list(payload.get("points", []))
            lines = list(payload.get("lines", []))
            dst_rect_override = payload.get("dst_rect_override")
        except Exception as exc:
            self._json_response({"error": f"Invalid request: {exc}"}, status=400)
            return

        image_path = TEST_IMAGES_DIR / image_name
        if not image_path.exists():
            self._json_response({"error": f"Image not found: {image_name}"}, status=404)
            return

        preview_dir = PREVIEW_DIR / image_path.stem
        try:
            result = build_tube_detection_preview(
                image_path=image_path,
                lines=lines,
                points=points,
                rois=rois,
                output_dir=preview_dir,
                dst_rect_override=dst_rect_override,
            )
        except Exception as exc:
            self._json_response({"error": str(exc)}, status=400)
            return

        hom = result.homography
        self._json_response(
            {
                "overlay_url": f"/artifacts/web_homography_preview/{image_path.stem}/{hom.overlay_path.name}",
                "warp_url": f"/artifacts/web_homography_preview/{image_path.stem}/{result.detection_overlay_path.name}",
                "used_line_labels": hom.used_line_labels,
                "src_points": hom.src_points,
                "output_size": hom.output_size,
                "base_size": hom.base_size,
                "warp_padding": hom.warp_padding,
                "dst_rect": hom.dst_rect,
                "homography_matrix": hom.homography_matrix,
                "inverse_homography_matrix": hom.inverse_homography_matrix,
                "tube_count": result.tube_count,
                "dominant_period": result.dominant_period,
                "energy_start_index": result.energy_start_index,
                "peaks_index": result.peaks_index,
                "peaks_index_dom": result.peaks_index_dom,
                "detection_roi": result.detection_roi,
                "x_start_list": result.x_start_list,
            }
        )


def main() -> None:
    host = "127.0.0.1"
    port = 8765
    server = ThreadingHTTPServer((host, port), RoiWebHandler)
    print(f"ROI web app running at http://{host}:{port}/web_roi_picker/")
    print(f"Serving project root: {ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
