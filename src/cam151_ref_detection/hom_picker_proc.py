"""
Homography point picker for notebooks and command-line workflows.
Call run_picker(image_bgr, existing_pts) to return a dict or None.
"""
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import tempfile
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import cv2
import numpy as np


_PICKER_SCRIPT = r"""
import sys, json, cv2, numpy as np
import tkinter as tk
from PIL import Image, ImageTk

data = json.loads(sys.argv[1])
img_path   = data['img_path']
existing   = data['existing']   # {label: [x,y]} or {}
max_win    = data.get('max_win', 820)
PREV_W, PREV_H = 340, 460

image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
assert image_bgr is not None
img_h, img_w = image_bgr.shape[:2]
pil_full = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

_H_LABELS = ('TL', 'TR', 'BL', 'BR')
_H_COLORS = {'TL': '#00ff44', 'TR': '#ffcc00', 'BL': '#ff4444', 'BR': '#44aaff'}
_H_NAMES  = {'TL': 'P1 arriba-izq', 'TR': 'P2 arriba-der', 'BL': 'P3 abajo-izq', 'BR': 'P4 abajo-der'}

def _warp_preview(pts_dict):
    if not all(lbl in pts_dict for lbl in _H_LABELS):
        return None
    src = np.float32([pts_dict[l] for l in _H_LABELS])
    dst = np.float32([[0,0],[PREV_W-1,0],[0,PREV_H-1],[PREV_W-1,PREV_H-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, M, (PREV_W, PREV_H))

state = {
    'zoom': min(1.0, max_win / max(img_w, img_h)),
    'offset_x': 0.0, 'offset_y': 0.0,
    'pts': {k: list(v) for k,v in existing.items()} if existing else {},
    'next_idx': 0,
    'drag_lbl': None,
    'pan_start': None, 'pan_off_start': None,
    'confirmed': False,
}
for i, lbl in enumerate(_H_LABELS):
    if lbl not in state['pts']:
        state['next_idx'] = i; break
else:
    state['next_idx'] = 0

win_w = min(img_w + 20, max_win)
win_h = min(img_h + 40, max_win)

root = tk.Tk()
root.title('Homografia | Click-izq=colocar  Click-der+drag=mover  Scroll=zoom  MidDrag=pan  DEL=borrar  Enter=OK  ESC=cancelar')
root.resizable(True, True)
root.lift()
root.attributes('-topmost', True)
root.after(300, lambda: root.attributes('-topmost', False))

left_frame = tk.Frame(root, bg='#111')
left_frame.pack(side='left', fill='both', expand=True)
canvas = tk.Canvas(left_frame, width=win_w, height=win_h, bg='#111', cursor='crosshair')
canvas.pack(fill='both', expand=True)
status = tk.Label(left_frame, text='', bg='#222', fg='lime', font=('Arial', 10), anchor='w', padx=6)
status.pack(fill='x')

right_frame = tk.Frame(root, bg='#1a1a1a', width=PREV_W + 20)
right_frame.pack(side='right', fill='y')
right_frame.pack_propagate(False)
tk.Label(right_frame, text='WARP PREVIEW', bg='#1a1a1a', fg='#aaa', font=('Arial',9,'bold')).pack(pady=(8,2))
tk.Label(right_frame, text='P1--P2\n|    |\nP3--P4', bg='#1a1a1a', fg='#666', font=('Courier',9)).pack(pady=(0,4))
prev_canvas = tk.Canvas(right_frame, width=PREV_W, height=PREV_H, bg='#333')
prev_canvas.pack(padx=10, pady=4)
_prev_ref = [None]
_tk_ref   = [None]

def _i2c(ix,iy): z=state['zoom']; return ix*z+state['offset_x'], iy*z+state['offset_y']
def _c2i(cx,cy): z=state['zoom']; return (cx-state['offset_x'])/z, (cy-state['offset_y'])/z

def _nearest(cx,cy,thresh=20):
    best,bd=None,float('inf')
    for lbl,pt in state['pts'].items():
        px,py=_i2c(pt[0],pt[1]); d=((cx-px)**2+(cy-py)**2)**0.5
        if d<bd: bd,best=d,lbl
    return best if bd<thresh else None

def _draw_preview():
    prev_canvas.delete('all')
    w=_warp_preview(state['pts'])
    if w is not None:
        p=Image.fromarray(cv2.cvtColor(w,cv2.COLOR_BGR2RGB))
        tk_p=ImageTk.PhotoImage(p); _prev_ref[0]=tk_p
        prev_canvas.create_image(0,0,anchor='nw',image=tk_p)
        cdata=[('P1',(2,2),'nw'),('P2',(PREV_W-2,2),'ne'),('P3',(2,PREV_H-2),'sw'),('P4',(PREV_W-2,PREV_H-2),'se')]
        cols=[_H_COLORS[l] for l in _H_LABELS]
        for (nm,pos,anc),col in zip(cdata,cols):
            prev_canvas.create_text(pos[0],pos[1],text=nm,fill=col,font=('Arial',9,'bold'),anchor=anc)
    else:
        prev_canvas.create_text(PREV_W//2,PREV_H//2,
            text=f"({len(state['pts'])}/4)\nP1=arriba-izq\nP2=arriba-der\nP3=abajo-izq\nP4=abajo-der",
            fill='#888',font=('Arial',10),justify='center')

def _redraw():
    canvas.delete('all')
    cw=canvas.winfo_width() or win_w; ch=canvas.winfo_height() or win_h
    z=state['zoom']
    ix0,iy0=_c2i(0,0); ix1,iy1=_c2i(cw,ch)
    sx0=max(0,int(ix0)); sy0=max(0,int(iy0))
    sx1=min(img_w,int(ix1)+2); sy1=min(img_h,int(iy1)+2)
    if sx1>sx0 and sy1>sy0:
        crop=pil_full.crop((sx0,sy0,sx1,sy1))
        nw=max(1,int((sx1-sx0)*z)); nh=max(1,int((sy1-sy0)*z))
        tk_img=ImageTk.PhotoImage(crop.resize((nw,nh),Image.LANCZOS))
        _tk_ref[0]=tk_img
        ox,oy=_i2c(sx0,sy0); canvas.create_image(int(ox),int(oy),anchor='nw',image=tk_img)
    pts=state['pts']
    if all(l in pts for l in _H_LABELS):
        tl=_i2c(*pts['TL']); tr=_i2c(*pts['TR']); bl=_i2c(*pts['BL']); br=_i2c(*pts['BR'])
        for a,b in [(tl,tr),(bl,br),(tl,bl),(tr,br)]:
            canvas.create_line(a[0],a[1],b[0],b[1],fill='#ffffff66',width=1,dash=(4,3))
    for lbl,pt in pts.items():
        cx,cy=_i2c(pt[0],pt[1]); col=_H_COLORS[lbl]; r=9
        canvas.create_line(cx-r,cy,cx+r,cy,fill=col,width=2)
        canvas.create_line(cx,cy-r,cx,cy+r,fill=col,width=2)
        canvas.create_oval(cx-5,cy-5,cx+5,cy+5,outline=col,width=2)
        canvas.create_text(cx+12,cy-10,text=_H_NAMES[lbl],fill=col,font=('Arial',9,'bold'),anchor='w')
    ni=state['next_idx']; nxt=_H_LABELS[ni]; nc=_H_COLORS[nxt]
    if len(pts)<4:
        msg=f"Click izq -> {_H_NAMES[nxt]}  ({len(pts)}/4)  DEL=borrar  Enter=OK"
    else:
        msg="4/4 OK  Click-der+drag=ajustar  Enter=guardar  ESC=cancelar"; nc='#00ff88'
    status.config(text=msg,fg=nc)
    _draw_preview()

def _left_click(e):
    ix,iy=_c2i(e.x,e.y)
    if 0<=ix<=img_w and 0<=iy<=img_h:
        lbl=_H_LABELS[state['next_idx']%4]
        state['pts'][lbl]=[ix,iy]
        state['next_idx']=(state['next_idx']+1)%4
    _redraw()

def _r_press(e): state['drag_lbl']=_nearest(e.x,e.y)
def _r_drag(e):
    if not state['drag_lbl']: return
    ix,iy=_c2i(e.x,e.y)
    state['pts'][state['drag_lbl']]=[max(0.0,min(float(img_w),ix)),max(0.0,min(float(img_h),iy))]
    _redraw()
def _r_release(e): state['drag_lbl']=None

def _delete(e):
    if state['pts']:
        prev=(state['next_idx']-1)%4; lbl=_H_LABELS[prev]
        state['pts'].pop(lbl,None); state['next_idx']=prev
    _redraw()

def _scroll(e):
    f=1.15 if (e.delta>0 or e.num==4) else 1/1.15
    wx,wy=_c2i(e.x,e.y); state['zoom']=max(0.05,min(30.0,state['zoom']*f))
    state['offset_x']=e.x-wx*state['zoom']; state['offset_y']=e.y-wy*state['zoom']; _redraw()

def _m_press(e): state['pan_start']=(e.x,e.y); state['pan_off_start']=(state['offset_x'],state['offset_y'])
def _m_drag(e):
    if not state['pan_start']: return
    state['offset_x']=state['pan_off_start'][0]+e.x-state['pan_start'][0]
    state['offset_y']=state['pan_off_start'][1]+e.y-state['pan_start'][1]; _redraw()
def _m_release(e): state['pan_start']=None
def _ok(e=None): state['confirmed']=True; root.destroy()
def _cancel(e=None): root.destroy()

canvas.bind('<Button-1>',_left_click)
canvas.bind('<Button-3>',_r_press); canvas.bind('<B3-Motion>',_r_drag); canvas.bind('<ButtonRelease-3>',_r_release)
canvas.bind('<Delete>',_delete)
canvas.bind('<MouseWheel>',_scroll); canvas.bind('<Button-4>',_scroll); canvas.bind('<Button-5>',_scroll)
canvas.bind('<Button-2>',_m_press); canvas.bind('<B2-Motion>',_m_drag); canvas.bind('<ButtonRelease-2>',_m_release)
root.bind('<Return>',_ok); root.bind('<KP_Enter>',_ok); root.bind('<Escape>',_cancel)
canvas.focus_set()
z0=state['zoom']; state['offset_x']=(win_w-img_w*z0)/2; state['offset_y']=(win_h-img_h*z0)/2
root.after(50,_redraw); root.mainloop()

if state['confirmed'] and len(state['pts'])==4:
    print(json.dumps({lbl: state['pts'][lbl] for lbl in _H_LABELS}))
else:
    print('null')
"""


_H_LABELS = ("TL", "TR", "BL", "BR")
_H_NAMES = {
    "TL": "P1 arriba-izq",
    "TR": "P2 arriba-der",
    "BL": "P3 abajo-izq",
    "BR": "P4 abajo-der",
}


def _normalize_existing_points(existing_pts: dict | None) -> dict[str, list[float]]:
    normalized: dict[str, list[float]] = {}
    for label in _H_LABELS:
        value = (existing_pts or {}).get(label)
        if value is None:
            continue
        try:
            normalized[label] = [float(value[0]), float(value[1])]
        except (TypeError, ValueError, IndexError):
            continue
    return normalized


def _running_in_jupyter() -> bool:
    try:
        from IPython import get_ipython
    except Exception:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _prepare_image_path(image_bgr: np.ndarray | None, img_path: str | None) -> tuple[str, str | None]:
    if img_path:
        path = Path(img_path)
        if path.exists():
            return str(path), None

    if image_bgr is None:
        raise ValueError("run_picker requiere image_bgr o img_path.")

    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    if not cv2.imwrite(temp_path, image_bgr):
        Path(temp_path).unlink(missing_ok=True)
        raise RuntimeError("No se pudo escribir la imagen temporal del picker.")
    return temp_path, temp_path


def _image_mime_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg", ".jfif"}:
        return "image/jpeg"
    return "application/octet-stream"


def _display_jupyter_link(url: str) -> None:
    if not _running_in_jupyter():
        return
    try:
        from IPython.display import HTML, display

        display(HTML(f'<a href="{url}" target="_blank">Abrir picker de homografia</a>'))
    except Exception:
        return


def _validate_result_points(payload: Any) -> dict[str, list[float]] | None:
    if not isinstance(payload, dict) or not payload.get("confirmed"):
        return None
    pts = payload.get("pts")
    if not isinstance(pts, dict):
        return None

    result: dict[str, list[float]] = {}
    for label in _H_LABELS:
        value = pts.get(label)
        if value is None:
            return None
        try:
            result[label] = [float(value[0]), float(value[1])]
        except (TypeError, ValueError, IndexError):
            return None
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

    html_path = Path(__file__).with_name("homography_picker.html")
    html_template = html_path.read_text(encoding="utf-8")
    initial_state = {
        "existing": existing_pts,
        "labels": list(_H_LABELS),
        "names": _H_NAMES,
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

            result_box["result"] = _validate_result_points(payload)
            done.set()
            self._send_json({"ok": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), PickerHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Picker web listo: {url}")
    print("La celda queda esperando hasta que presiones Confirmar o Cancelar.")
    _display_jupyter_link(url)
    try:
        webbrowser.open(url, new=1)
    except Exception:
        pass

    try:
        if not done.wait(timeout=timeout):
            print("Picker web: tiempo agotado sin resultado.")
            return None
        return result_box["result"]
    except KeyboardInterrupt:
        print("Picker web cancelado desde el kernel.")
        return None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _run_subprocess_picker(image_path: str, existing_pts: dict[str, list[float]]) -> dict | None:
    payload = json.dumps({
        "img_path": str(image_path),
        "existing": existing_pts,
        "max_win": 820,
    })

    script_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    script_tmp.write(_PICKER_SCRIPT)
    script_tmp.close()

    try:
        result = subprocess.run(
            [sys.executable, script_tmp.name, payload],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            print("Picker error:", result.stderr[-2000:] if result.stderr else "(no stderr)")
            return None
        out = result.stdout.strip().splitlines()
        last = out[-1] if out else "null"
        if last == "null":
            return None
        return json.loads(last)
    finally:
        Path(script_tmp.name).unlink(missing_ok=True)


def _mp_picker_entry(payload: dict, result_queue: Any) -> None:
    try:
        old_argv = sys.argv[:]
        sys.argv = ["hom_picker_mp", json.dumps(payload)]
        try:
            import contextlib
            import io

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(_PICKER_SCRIPT, {"__name__": "__main__"})
            out = buffer.getvalue().strip().splitlines()
            last = out[-1] if out else "null"
            result = None if last == "null" else json.loads(last)
            result_queue.put({"ok": True, "result": result})
        finally:
            sys.argv = old_argv
    except BaseException:
        result_queue.put({"ok": False, "error": traceback.format_exc()})


def _run_multiprocessing_picker(
    image_path: str,
    existing_pts: dict[str, list[float]],
    *,
    timeout: float | None = None,
) -> dict | None:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    payload = {
        "img_path": str(image_path),
        "existing": existing_pts,
        "max_win": 820,
    }
    process = ctx.Process(target=_mp_picker_entry, args=(payload, result_queue), daemon=False)
    process.start()

    elapsed = 0.0
    try:
        while True:
            try:
                message = result_queue.get(timeout=0.2)
                break
            except queue.Empty:
                if timeout is not None:
                    elapsed += 0.2
                    if elapsed >= timeout:
                        process.terminate()
                        process.join(timeout=2)
                        print("Picker Tk: tiempo agotado sin resultado.")
                        return None
                if not process.is_alive():
                    process.join(timeout=2)
                    print(f"Picker Tk termino sin resultado. exitcode={process.exitcode}")
                    return None
    except KeyboardInterrupt:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        print("Picker Tk cancelado desde el kernel.")
        return None

    process.join(timeout=2)
    if not message.get("ok"):
        print("Picker Tk error:")
        print(str(message.get("error", ""))[-2000:])
        return None
    return message.get("result")


def run_picker(image_bgr: np.ndarray, existing_pts: dict | None = None,
               img_path: str | None = None, *, backend: str = "auto",
               timeout: float | None = None) -> dict | None:
    """
    Run the homography picker.

    backends:
      - auto: web inside Jupyter, multiprocessing Tk elsewhere.
      - web: local browser picker. This is the reliable path for classic
        Jupyter Notebook on Windows.
      - multiprocessing/process/tk: Tkinter in a spawned Python process.
      - subprocess: legacy temp-script launcher.

    Returns {TL:[x,y], TR:[x,y], BL:[x,y], BR:[x,y]} or None if cancelled.
    """
    normalized_existing = _normalize_existing_points(existing_pts)
    prepared_path, temp_path = _prepare_image_path(image_bgr, img_path)
    backend_key = str(backend or "auto").strip().lower()
    if backend_key == "auto":
        backend_key = "web" if _running_in_jupyter() else "multiprocessing"

    try:
        if backend_key in {"web", "browser", "http"}:
            return _run_web_picker(prepared_path, normalized_existing, timeout=timeout)
        if backend_key in {"multiprocessing", "process", "tk"}:
            return _run_multiprocessing_picker(prepared_path, normalized_existing, timeout=timeout)
        if backend_key == "subprocess":
            return _run_subprocess_picker(prepared_path, normalized_existing)
        raise ValueError(f"Backend de picker no soportado: {backend!r}")
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
