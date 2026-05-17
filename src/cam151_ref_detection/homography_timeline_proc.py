from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

import cv2

from .roi_store import load_rois


_ASSET_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass
class HomographyTimelineServerHandle:
    url: str
    server: ThreadingHTTPServer
    thread: threading.Thread

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def wait(self) -> None:
        try:
            self.thread.join()
        except KeyboardInterrupt:
            self.close()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_side(value: Any) -> str:
    raw = str(value or "152").strip().lower()
    if raw in {"151", "cam151", "cam_151", "left"}:
        return "151"
    if raw in {"152", "cam152", "cam_152", "right"}:
        return "152"
    return "152"


def _camera_key(side: str) -> str:
    return f"cam{_normalize_side(side)}"


def _resolve_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    candidate = _repo_root() / path
    if candidate.exists():
        return candidate
    return None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _image_dimensions(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return (0, 0)
    height, width = image.shape[:2]
    return int(width), int(height)


def _image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _asset_token(*parts: Any) -> str:
    joined = "_".join(str(part or "") for part in parts)
    token = _ASSET_TOKEN_RE.sub("_", joined).strip("_")
    return token or "asset"


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_dt(value: Any) -> str:
    dt = _parse_dt(value)
    if dt is None:
        return str(value or "")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _hour_label(value: Any) -> str:
    dt = _parse_dt(value)
    if dt is None:
        return "No time"
    return dt.strftime("%Y-%m-%d %H:00")


def _default_roi_path(side: str) -> Path:
    side_key = _normalize_side(side)
    return _repo_root() / "manual_rois" / "_current_defaults" / f"cam_{side_key}_current_default_rois.toml"


def _default_image_path(side: str) -> Path:
    side_key = _normalize_side(side)
    return _repo_root() / "test_images" / f"cam_{side_key}_202604022.jpeg"


def _baseline_artifact_path(side: str, name: str) -> Path | None:
    side_key = _normalize_side(side)
    candidate = (
        _repo_root()
        / "artifacts"
        / f"notebook_step_by_step_cam{side_key}"
        / f"cam_{side_key}_202604022"
        / "01_homography"
        / name
    )
    return candidate if candidate.exists() else None


def _capture_runs_dir() -> Path:
    return _repo_root() / "pipe_end_detection" / "captures" / "runs"


def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, str]:
    if entry.get("kind") == "default":
        return (0, "")
    return (1, str(entry.get("captured_at") or entry.get("run_id") or ""))


def _load_default_points(side: str) -> tuple[Path, list[list[float]], list[float] | None]:
    roi_path = _default_roi_path(side)
    if not roi_path.exists():
        raise FileNotFoundError(f"Missing default ROI for cam{side}: {roi_path}")
    payload = load_rois(roi_path)
    points = payload.get("src_points_override")
    if not isinstance(points, list) or len(points) != 4:
        raise RuntimeError(f"Default ROI has no valid src_points_override: {roi_path}")
    normalized_points = [[float(point[0]), float(point[1])] for point in points]
    dst_rect = payload.get("dst_rect_override")
    normalized_rect = [float(value) for value in dst_rect] if isinstance(dst_rect, list) and len(dst_rect) == 4 else None
    return roi_path, normalized_points, normalized_rect


def _build_baseline_entry(side: str) -> dict[str, Any] | None:
    side_key = _normalize_side(side)
    raw_path = _default_image_path(side_key)
    if not raw_path.exists():
        return None
    return {
        "kind": "default",
        "side": side_key,
        "run_id": "default_20260422",
        "captured_at": "2026-04-22T00:00:00-06:00",
        "display_time": "Default 2026-04-22",
        "hour_label": "Default baseline",
        "image_name": raw_path.name,
        "raw_path": raw_path,
        "warp_path": _baseline_artifact_path(side_key, "homography_warp.jpg"),
        "overlay_path": _baseline_artifact_path(side_key, "homography_overlay.jpg"),
        "status": "default",
        "ptz": {},
    }


def _build_capture_entry(manifest_path: Path, side: str) -> dict[str, Any] | None:
    manifest = _read_json(manifest_path)
    side_key = _normalize_side(side)
    camera_key = _camera_key(side_key)
    camera_info = dict((manifest.get("cameras") or {}).get(camera_key) or {})
    raw_path = _resolve_path(camera_info.get("image_path"))
    if raw_path is None or not raw_path.exists():
        return None

    homography = dict((manifest.get("homography") or {}).get(camera_key) or {})
    captured_at = str(camera_info.get("captured_at") or manifest.get("captured_at") or "")
    run_id = str(manifest.get("run_id") or manifest_path.parent.name)
    return {
        "kind": "capture",
        "side": side_key,
        "run_id": run_id,
        "captured_at": captured_at,
        "display_time": _format_dt(captured_at),
        "hour_label": _hour_label(captured_at),
        "image_name": str(camera_info.get("image_name") or raw_path.name),
        "raw_path": raw_path,
        "warp_path": _resolve_path(homography.get("warp_path")),
        "overlay_path": _resolve_path(homography.get("overlay_path")),
        "status": str(manifest.get("status") or ""),
        "ptz": dict(camera_info.get("ptz") or {}),
    }


def _attach_assets(entry: dict[str, Any], asset_map: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    enriched = dict(entry)
    for field in ("raw_path", "warp_path", "overlay_path"):
        path = enriched.pop(field, None)
        url_key = field.replace("_path", "_url")
        enriched[url_key] = None
        if not path:
            continue
        path_obj = Path(path)
        if not path_obj.exists():
            continue
        token = _asset_token(enriched.get("run_id"), enriched.get("side"), field, path_obj.name)
        url = f"/asset/{token}{path_obj.suffix.lower()}"
        asset_map[url] = (path_obj, _image_mime_type(path_obj))
        enriched[url_key] = url
        if field == "raw_path":
            width, height = _image_dimensions(path_obj)
            enriched["raw_width"] = width
            enriched["raw_height"] = height
        if field == "warp_path":
            width, height = _image_dimensions(path_obj)
            enriched["warp_width"] = width
            enriched["warp_height"] = height
    return enriched


def _build_state(side: str, asset_map: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    side_key = _normalize_side(side)
    roi_path, default_points, dst_rect = _load_default_points(side_key)

    entries: list[dict[str, Any]] = []
    baseline = _build_baseline_entry(side_key)
    if baseline is not None:
        entries.append(baseline)

    runs_dir = _capture_runs_dir()
    if runs_dir.exists():
        for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
            entry = _build_capture_entry(manifest_path, side_key)
            if entry is not None:
                entries.append(entry)

    entries.sort(key=_entry_sort_key)
    asset_map.clear()
    enriched_entries = [_attach_assets(entry, asset_map) for entry in entries]
    return {
        "camera": side_key,
        "camera_label": f"cam{side_key}",
        "default_roi_path": str(roi_path),
        "default_points": default_points,
        "default_dst_rect": dst_rect,
        "entries": enriched_entries,
        "counts": {
            "entries": len(enriched_entries),
            "captures": max(0, len(enriched_entries) - (1 if baseline is not None else 0)),
        },
    }


def _timeline_html(state: dict[str, Any]) -> bytes:
    payload = json.dumps(state, ensure_ascii=False)
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Homography Timeline</title>
<style>
:root {
  --ink:#18140c;
  --muted:#635b49;
  --panel:#fffaf0;
  --line:#d7c89f;
  --shadow:0 18px 48px rgba(60,43,9,.14);
}
* { box-sizing:border-box; }
body {
  margin:0;
  font-family: Georgia, "Times New Roman", serif;
  color:var(--ink);
  background:
    radial-gradient(circle at 12% 10%, rgba(255,255,255,.95), transparent 34rem),
    linear-gradient(135deg, #f9f1d4 0%, #f4e5b6 42%, #e8d6a2 100%);
}
header {
  position:sticky;
  top:0;
  z-index:20;
  padding:14px 18px;
  border-bottom:1px solid rgba(91,65,12,.22);
  background:rgba(247,242,223,.94);
  backdrop-filter:blur(12px);
}
.topline {
  display:flex;
  gap:16px;
  align-items:center;
  justify-content:space-between;
  flex-wrap:wrap;
}
h1 {
  margin:0;
  font-size:clamp(24px, 3vw, 40px);
  letter-spacing:-.05em;
}
.subtitle {
  margin:4px 0 0;
  color:var(--muted);
  max-width:850px;
  line-height:1.35;
}
.controls {
  display:flex;
  gap:9px;
  align-items:center;
  flex-wrap:wrap;
}
.pill, button, select {
  border:1px solid rgba(91,65,12,.34);
  background:#fff8e7;
  border-radius:999px;
  padding:10px 14px;
  color:#2b210c;
  box-shadow:0 8px 18px rgba(91,65,12,.08);
}
button {
  cursor:pointer;
  font-weight:700;
}
button:disabled {
  opacity:.45;
  cursor:not-allowed;
}
button.active {
  color:#fff;
  background:#553d08;
}
select {
  min-width:min(600px, 82vw);
  border-radius:14px;
}
main {
  padding:16px;
}
.panel {
  border:1px solid rgba(91,65,12,.22);
  border-radius:20px;
  background:rgba(255,250,240,.9);
  box-shadow:var(--shadow);
  padding:12px;
}
.meta {
  margin-bottom:14px;
}
.meta h2 {
  margin:0 0 6px;
  font-size:24px;
  letter-spacing:-.03em;
}
.meta-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
  gap:6px 14px;
  color:var(--muted);
  font-size:13px;
}
.viewer-grid {
  display:grid;
  grid-template-columns:minmax(0, 1.15fr) minmax(360px, .85fr);
  gap:16px;
  align-items:start;
}
.panel h3 {
  margin:0 0 8px;
  font-size:18px;
}
.zoom-frame {
  position:relative;
  overflow:hidden;
  border-radius:14px;
  border:1px solid rgba(91,65,12,.24);
  background:#111;
  min-height:360px;
  cursor:grab;
  touch-action:none;
}
.zoom-frame:active {
  cursor:grabbing;
}
.zoom-content {
  position:relative;
  transform-origin:0 0;
  will-change:transform;
}
.zoom-content img {
  display:block;
  width:100%;
  height:auto;
  user-select:none;
  -webkit-user-drag:none;
}
.zoom-content img.mirror-x {
  transform:scaleX(-1);
}
.zoom-content svg {
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  pointer-events:none;
}
.frame-label {
  position:absolute;
  left:8px;
  top:8px;
  padding:4px 7px;
  border-radius:999px;
  background:rgba(0,0,0,.68);
  color:white;
  font-size:11px;
  font-family:Arial, sans-serif;
}
.missing {
  height:360px;
  display:grid;
  place-items:center;
  color:#c9bd98;
  font:12px Arial, sans-serif;
}
.side-stack {
  display:grid;
  gap:12px;
}
.point {
  fill:#ffce00;
  stroke:#111;
  stroke-width:5;
  vector-effect:non-scaling-stroke;
}
.quad {
  fill:rgba(255,206,0,.08);
  stroke:#ffce00;
  stroke-width:4;
  vector-effect:non-scaling-stroke;
}
.small {
  font-size:12px;
  color:var(--muted);
  word-break:break-all;
}
@media (max-width: 1100px) {
  .viewer-grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<header>
  <div class="topline">
    <div>
      <h1>Homography Timeline</h1>
      <p class="subtitle">One capture at a time. Use Next or the date selector. Wheel zooms; drag pans. The main image is the saved homography overlay generated by the backend.</p>
    </div>
    <div class="controls">
      <button id="cam151">cam151</button>
      <button id="cam152">cam152</button>
      <button id="prevBtn">Previous</button>
      <button id="nextBtn">Next</button>
      <select id="runSelect"></select>
      <button id="resetZoomBtn">Reset zoom</button>
      <span class="pill" id="countPill"></span>
    </div>
  </div>
</header>
<main>
  <section id="metaPanel" class="panel meta"></section>
  <section class="viewer-grid">
    <div class="panel" id="rawPanel"></div>
    <div class="panel side-stack" id="sidePanel"></div>
  </section>
</main>
<script>
const STATE = __STATE__;
const points = STATE.default_points || [];
let selectedIndex = 0;
const zoomState = new Map();
const metaPanel = document.getElementById("metaPanel");
const rawPanel = document.getElementById("rawPanel");
const sidePanel = document.getElementById("sidePanel");
const runSelect = document.getElementById("runSelect");

document.getElementById("cam151").classList.toggle("active", STATE.camera === "151");
document.getElementById("cam152").classList.toggle("active", STATE.camera === "152");
document.getElementById("cam151").onclick = () => location.href = "?camera=151";
document.getElementById("cam152").onclick = () => location.href = "?camera=152";
document.getElementById("prevBtn").onclick = () => move(-1);
document.getElementById("nextBtn").onclick = () => move(1);
document.getElementById("resetZoomBtn").onclick = () => resetAllZoom();
runSelect.onchange = () => selectIndex(Number(runSelect.value || 0));

function pointMarkup(entry) {
  const w = Number(entry.raw_width || 2048);
  const h = Number(entry.raw_height || 1536);
  const mirrorPoints = STATE.camera === "152";
  let mapped = points.map(p => [mirrorPoints ? (w - 1 - Number(p[0])) : Number(p[0]), Number(p[1])]);
  if(mirrorPoints && mapped.length === 4) {
    mapped = [mapped[1], mapped[0], mapped[3], mapped[2]];
  }
  const polygonPoints = mapped.length === 4 ? [mapped[0], mapped[1], mapped[3], mapped[2]] : mapped;
  const poly = polygonPoints.map(p => `${p[0]},${p[1]}`).join(" ");
  const circles = mapped.map((p, i) => `<circle class="point" cx="${p[0]}" cy="${p[1]}" r="${i === 0 ? 11 : 9}"></circle>`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet"><polygon class="quad" points="${poly}"></polygon>${circles}</svg>`;
}

function zoomFrame(id, url, label, overlay="", mirrorImage=false) {
  if(!url) return `<div class="zoom-frame"><div class="missing">missing ${label}</div></div>`;
  return `<div class="zoom-frame" data-zoom-id="${id}">
    <div class="zoom-content">
      <img class="${mirrorImage ? "mirror-x" : ""}" src="${url}" alt="${label}" draggable="false">
      ${overlay}
    </div>
    <div class="frame-label">${label}</div>
  </div>`;
}

function renderOptions() {
  runSelect.innerHTML = STATE.entries.map((entry, index) => {
    const label = `${String(index + 1).padStart(3, "0")} - ${entry.display_time || entry.hour_label || ""} - ${entry.run_id}`;
    return `<option value="${index}">${label}</option>`;
  }).join("");
}

function currentEntry() {
  return STATE.entries[Math.max(0, Math.min(selectedIndex, STATE.entries.length - 1))];
}

function selectIndex(index) {
  selectedIndex = Math.max(0, Math.min(index, STATE.entries.length - 1));
  render();
}

function move(delta) {
  selectIndex(selectedIndex + delta);
}

function render() {
  const entry = currentEntry();
  if(!entry) return;
  runSelect.value = String(selectedIndex);
  document.getElementById("countPill").textContent = `${STATE.camera_label} - ${selectedIndex + 1} / ${STATE.entries.length}`;
  document.getElementById("prevBtn").disabled = selectedIndex <= 0;
  document.getElementById("nextBtn").disabled = selectedIndex >= STATE.entries.length - 1;
  const rawLabel = "saved homography overlay";

  metaPanel.innerHTML = `<h2>${entry.run_id}</h2>
    <div class="meta-grid">
      <div><strong>${entry.display_time || ""}</strong></div>
      <div>${entry.image_name || ""}</div>
      <div>Hour: ${entry.hour_label || "-"}</div>
      <div>Status: ${entry.status || "-"}</div>
      <div>PTZ: ${JSON.stringify(entry.ptz || {})}</div>
      <div class="small">Default ROI: ${STATE.default_roi_path}</div>
    </div>`;
  rawPanel.innerHTML = `<h3>${rawLabel}</h3>${zoomFrame("overlay", entry.overlay_url || entry.raw_url, rawLabel)}`;
  sidePanel.innerHTML = `<div><h3>Warp</h3>${zoomFrame("warp", entry.warp_url, "warp")}</div>
    <div><h3>Raw capture</h3>${zoomFrame("raw", entry.raw_url, "raw capture")}</div>`;
  attachZoomHandlers();
}

function getZoom(id) {
  if(!zoomState.has(id)) zoomState.set(id, {scale:1, tx:0, ty:0});
  return zoomState.get(id);
}

function applyZoom(frame) {
  const id = frame.dataset.zoomId;
  const state = getZoom(id);
  const content = frame.querySelector(".zoom-content");
  if(content) content.style.transform = `translate(${state.tx}px, ${state.ty}px) scale(${state.scale})`;
}

function resetAllZoom() {
  zoomState.clear();
  document.querySelectorAll(".zoom-frame").forEach(applyZoom);
}

function attachZoomHandlers() {
  document.querySelectorAll(".zoom-frame").forEach(frame => {
    applyZoom(frame);
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    frame.addEventListener("wheel", event => {
      event.preventDefault();
      const id = frame.dataset.zoomId;
      const state = getZoom(id);
      const rect = frame.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const oldScale = state.scale;
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      const nextScale = Math.max(0.5, Math.min(8, oldScale * factor));
      state.tx = px - ((px - state.tx) * nextScale / oldScale);
      state.ty = py - ((py - state.ty) * nextScale / oldScale);
      state.scale = nextScale;
      applyZoom(frame);
    }, {passive:false});
    frame.addEventListener("pointerdown", event => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      frame.setPointerCapture(event.pointerId);
    });
    frame.addEventListener("pointermove", event => {
      if(!dragging) return;
      const state = getZoom(frame.dataset.zoomId);
      state.tx += event.clientX - lastX;
      state.ty += event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      applyZoom(frame);
    });
    frame.addEventListener("pointerup", () => dragging = false);
    frame.addEventListener("pointercancel", () => dragging = false);
  });
}

document.addEventListener("keydown", event => {
  if(event.key === "ArrowRight") move(1);
  if(event.key === "ArrowLeft") move(-1);
});

renderOptions();
render();
</script>
</body>
</html>"""
    return html.replace("__STATE__", payload).encode("utf-8")


def start_homography_timeline_server(
    *,
    camera: str = "152",
    port: int | None = None,
    open_browser: bool = True,
    on_ready: Any | None = None,
) -> HomographyTimelineServerHandle:
    side = _normalize_side(camera)
    asset_map: dict[str, tuple[Path, str]] = {}

    class TimelineHandler(BaseHTTPRequestHandler):
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
            self._send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                query_side = _normalize_side((parse_qs(parsed.query).get("camera") or [side])[0])
                try:
                    state = _build_state(query_side, asset_map)
                    self._send_bytes(_timeline_html(state), "text/html; charset=utf-8")
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)
                return
            if parsed.path.startswith("/asset/"):
                asset = asset_map.get(parsed.path)
                if asset is None:
                    self._send_json({"error": "asset not found"}, status=404)
                    return
                path, mime_type = asset
                self._send_bytes(path.read_bytes(), mime_type)
                return
            self._send_json({"error": "not found"}, status=404)

    server = ThreadingHTTPServer(("127.0.0.1", int(port or 0)), TimelineHandler)
    host, bound_port = server.server_address
    url = f"http://{host}:{bound_port}/?camera={side}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Homography timeline ready: {url}")
    if on_ready is not None:
        on_ready(url)
    if open_browser:
        try:
            webbrowser.open(url, new=1)
        except Exception:
            pass
    return HomographyTimelineServerHandle(url=url, server=server, thread=thread)
