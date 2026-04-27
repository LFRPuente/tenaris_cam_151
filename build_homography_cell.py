"""Insert homography point picker cell into both cam151 and cam152 notebooks."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).parent

MD_SRC = (
    "## 1b. Definir Puntos de Homografía (Tkinter)\n\n"
    "Coloca los **4 puntos** sobre la imagen original que definen la homografía.\n"
    "Se guardan como `src_points_override` en el TOML.\n\n"
    "**Orden de los puntos:**\n"
    "1. **TL** — top-left (esquina superior izquierda del paquete)\n"
    "2. **TR** — top-right (esquina superior derecha)\n"
    "3. **BL** — bottom-left (esquina inferior izquierda)\n"
    "4. **BR** — bottom-right (esquina inferior derecha)\n\n"
    "**Controles:**\n"
    "- **Click izquierdo** → colocar punto en orden TL→TR→BL→BR (se repite al completar)\n"
    "- **Scroll** → zoom centrado en el cursor\n"
    "- **Click medio + arrastrar** → paneo\n"
    "- **DEL** → borrar el último punto\n"
    "- **Enter** → confirmar y guardar en el TOML\n"
    "- **ESC** → cancelar sin guardar\n"
)

CODE_LINES = [
    "# ── Picker de puntos de homografía (Tkinter) ─────────────────────────────────",
    "# Coloca TL, TR, BL, BR sobre la imagen original.",
    "# Guarda src_points_override en el TOML y actualiza homography_src_points.",
    "",
    "import tkinter as tk",
    "from PIL import Image, ImageTk",
    "from src.cam151_ref_detection.roi_store import load_rois",
    "from src.cam151_ref_detection.roi_store import _to_toml as _roi_to_toml",
    "",
    "_H_LABELS  = ('TL', 'TR', 'BL', 'BR')",
    "_H_COLORS  = {'TL': '#00ff44', 'TR': '#ffcc00', 'BL': '#ff4444', 'BR': '#44aaff'}",
    "_H_DESCR   = {'TL': 'superior izq', 'TR': 'superior der', 'BL': 'inferior izq', 'BR': 'inferior der'}",
    "",
    "# Leer src_points_override actual del TOML.",
    "_existing_src = data.get('src_points_override')  # [[x,y],[x,y],[x,y],[x,y]] o None",
    "if _existing_src and len(_existing_src) == 4:",
    "    _pts_init = {lbl: list(xy) for lbl, xy in zip(_H_LABELS, _existing_src)}",
    "    print('src_points_override actuales:')",
    "    for lbl, xy in _pts_init.items():",
    "        print(f'  {lbl}: {xy}')",
    "else:",
    "    _pts_init = {}",
    "    print('No hay src_points_override definidos.')",
    "",
    "def _pick_homography_points_tk(image_bgr, existing_pts=None, max_win=900):",
    '    """',
    "    existing_pts: dict {label: [x,y]} con las posiciones actuales.",
    "    Devuelve dict {label: [x,y]} con los 4 puntos, o None si canceló.",
    '    """',
    "    img_h, img_w = image_bgr.shape[:2]",
    "    pil_full = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))",
    "",
    "    state = {",
    "        'zoom': min(1.0, max_win / max(img_w, img_h)),",
    "        'offset_x': 0.0, 'offset_y': 0.0,",
    "        'pts': dict(existing_pts) if existing_pts else {},",
    "        'next_idx': 0,",
    "        'pan_start': None, 'pan_off_start': None,",
    "        'confirmed': False,",
    "    }",
    "    # Apuntar next_idx al primer label no definido",
    "    for i, lbl in enumerate(_H_LABELS):",
    "        if lbl not in state['pts']:",
    "            state['next_idx'] = i",
    "            break",
    "    else:",
    "        state['next_idx'] = 0",
    "",
    "    win_w = min(img_w + 20, max_win)",
    "    win_h = min(img_h + 40, max_win)",
    "    root = tk.Tk()",
    "    root.title('Homography points  |  Click=colocar  Scroll=zoom  MidDrag=pan  DEL=borrar  Enter=OK  ESC=cancelar')",
    "    root.resizable(True, True)",
    "    canvas = tk.Canvas(root, width=win_w, height=win_h, bg='#111', cursor='crosshair')",
    "    canvas.pack(fill='both', expand=True)",
    "    status = tk.Label(root, text='', bg='#222', fg='lime', font=('Arial', 10), anchor='w', padx=6)",
    "    status.pack(fill='x')",
    "    _tk_img_ref = [None]",
    "",
    "    def _i2c(ix,iy): z=state['zoom']; return ix*z+state['offset_x'], iy*z+state['offset_y']",
    "    def _c2i(cx,cy): z=state['zoom']; return (cx-state['offset_x'])/z, (cy-state['offset_y'])/z",
    "",
    "    def _redraw():",
    "        canvas.delete('all')",
    "        cw=canvas.winfo_width() or win_w; ch=canvas.winfo_height() or win_h",
    "        z=state['zoom']",
    "        ix0,iy0=_c2i(0,0); ix1,iy1=_c2i(cw,ch)",
    "        sx0=max(0,int(ix0)); sy0=max(0,int(iy0))",
    "        sx1=min(img_w,int(ix1)+2); sy1=min(img_h,int(iy1)+2)",
    "        if sx1>sx0 and sy1>sy0:",
    "            crop=pil_full.crop((sx0,sy0,sx1,sy1))",
    "            nw=max(1,int((sx1-sx0)*z)); nh=max(1,int((sy1-sy0)*z))",
    "            tk_img=ImageTk.PhotoImage(crop.resize((nw,nh),Image.LANCZOS))",
    "            _tk_img_ref[0]=tk_img",
    "            ox,oy=_i2c(sx0,sy0); canvas.create_image(int(ox),int(oy),anchor='nw',image=tk_img)",
    "        # Dibujar líneas entre puntos si hay 4",
    "        pts = state['pts']",
    "        if all(lbl in pts for lbl in _H_LABELS):",
    "            tl=_i2c(*pts['TL']); tr=_i2c(*pts['TR'])",
    "            bl=_i2c(*pts['BL']); br=_i2c(*pts['BR'])",
    "            for a,b in [(tl,tr),(bl,br),(tl,bl),(tr,br)]:",
    "                canvas.create_line(a[0],a[1],b[0],b[1], fill='#ffffff', width=1, dash=(4,3))",
    "        # Dibujar puntos",
    "        for lbl, (ix,iy) in pts.items():",
    "            cx,cy=_i2c(ix,iy); color=_H_COLORS[lbl]; r=9",
    "            canvas.create_line(cx-r,cy,cx+r,cy, fill=color, width=2)",
    "            canvas.create_line(cx,cy-r,cx,cy+r, fill=color, width=2)",
    "            canvas.create_oval(cx-4,cy-4,cx+4,cy+4, outline=color, width=2)",
    "            canvas.create_text(cx+12,cy-10, text=f'{lbl} ({_H_DESCR[lbl]})',",
    "                               fill=color, font=('Arial',9,'bold'), anchor='w')",
    "        ni=state['next_idx']",
    "        nxt=_H_LABELS[ni]; nc=_H_COLORS[nxt]",
    "        status.config(",
    "            text=f'Click → {nxt} ({_H_DESCR[nxt]})   {len(pts)}/4   DEL=borrar último   Enter=OK   ESC=cancelar',",
    "            fg=nc)",
    "",
    "    def _on_click(e):",
    "        ix,iy=_c2i(e.x,e.y)",
    "        if 0<=ix<=img_w and 0<=iy<=img_h:",
    "            lbl=_H_LABELS[state['next_idx']%4]",
    "            state['pts'][lbl]=[ix,iy]",
    "            state['next_idx']=(state['next_idx']+1)%4",
    "        _redraw()",
    "",
    "    def _on_delete(e):",
    "        if state['pts']:",
    "            prev=(state['next_idx']-1)%4",
    "            lbl=_H_LABELS[prev]",
    "            state['pts'].pop(lbl,None)",
    "            state['next_idx']=prev",
    "        _redraw()",
    "",
    "    def _on_scroll(e):",
    "        f=1.15 if (e.delta>0 or e.num==4) else 1/1.15",
    "        wx,wy=_c2i(e.x,e.y); state['zoom']=max(0.05,min(30.0,state['zoom']*f))",
    "        state['offset_x']=e.x-wx*state['zoom']; state['offset_y']=e.y-wy*state['zoom']; _redraw()",
    "",
    "    def _on_mid_press(e):",
    "        state['pan_start']=(e.x,e.y); state['pan_off_start']=(state['offset_x'],state['offset_y'])",
    "    def _on_mid_drag(e):",
    "        if state['pan_start'] is None: return",
    "        state['offset_x']=state['pan_off_start'][0]+e.x-state['pan_start'][0]",
    "        state['offset_y']=state['pan_off_start'][1]+e.y-state['pan_start'][1]; _redraw()",
    "    def _on_mid_release(e): state['pan_start']=None",
    "    def _on_confirm(e=None): state['confirmed']=True; root.destroy()",
    "    def _on_cancel(e=None): root.destroy()",
    "",
    "    canvas.bind('<Button-1>',_on_click)",
    "    canvas.bind('<Delete>',_on_delete)",
    "    canvas.bind('<MouseWheel>',_on_scroll); canvas.bind('<Button-4>',_on_scroll); canvas.bind('<Button-5>',_on_scroll)",
    "    canvas.bind('<Button-2>',_on_mid_press); canvas.bind('<B2-Motion>',_on_mid_drag)",
    "    canvas.bind('<ButtonRelease-2>',_on_mid_release)",
    "    root.bind('<Return>',_on_confirm); root.bind('<KP_Enter>',_on_confirm); root.bind('<Escape>',_on_cancel)",
    "    canvas.focus_set()",
    "    z0=state['zoom']; state['offset_x']=(win_w-img_w*z0)/2; state['offset_y']=(win_h-img_h*z0)/2",
    "    root.after(50,_redraw); root.mainloop()",
    "    if state['confirmed'] and len(state['pts'])==4:",
    "        return state['pts']",
    "    return None",
    "",
    "print('Abriendo ventana para colocar los 4 puntos de homografía...')",
    "_result_pts = _pick_homography_points_tk(processing_image_bgr, _pts_init)",
    "",
    "if _result_pts is not None:",
    "    # Guardar en TOML como [[TL],[TR],[BL],[BR]]",
    "    _new_src = [[round(float(_result_pts[lbl][0]),3), round(float(_result_pts[lbl][1]),3)]",
    "                for lbl in _H_LABELS]",
    "    _roi_data_now = load_rois(ROI_PATH)",
    "    _roi_data_now['src_points_override'] = _new_src",
    "    ROI_PATH.write_text(_roi_to_toml(_roi_data_now), encoding='utf-8')",
    "    data['src_points_override'] = _new_src",
    "    homography_src_points = _new_src",
    "    print('src_points_override guardado:')",
    "    for lbl, xy in zip(_H_LABELS, _new_src):",
    "        print(f'  {lbl}: {xy}')",
    "    # Mostrar sobre la imagen",
    "    _vis = processing_image_bgr.copy()",
    "    _colors_cv = {'TL':(0,255,68),'TR':(0,204,255),'BL':(68,68,255),'BR':(255,170,0)}",
    "    for lbl, (ix,iy) in _result_pts.items():",
    "        cx,cy=int(round(ix)),int(round(iy)); color=_colors_cv[lbl]",
    "        cv2.circle(_vis,(cx,cy),8,color,-1,cv2.LINE_AA)",
    "        cv2.putText(_vis,lbl,(cx+10,cy-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)",
    "    pts_ord=[_result_pts[l] for l in _H_LABELS]",
    "    for a,b in [(0,1),(2,3),(0,2),(1,3)]:",
    "        p1=(int(round(pts_ord[a][0])),int(round(pts_ord[a][1])))",
    "        p2=(int(round(pts_ord[b][0])),int(round(pts_ord[b][1])))",
    "        cv2.line(_vis,p1,p2,(200,200,200),1,cv2.LINE_AA)",
    "    show_bgr(_vis,'Puntos de homografia seleccionados (TL/TR/BL/BR)',figsize=(8,12))",
    "    print('Vuelve a correr la celda de homografia (celda 3) para aplicar los nuevos puntos.')",
    "else:",
    "    print('Sin cambios — src_points_override no modificado.')",
]

CODE_SRC = "\n".join(CODE_LINES) + "\n"
ast.parse(CODE_SRC)
print("Syntax OK")


def make_md(source, cell_id):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}


def make_code(source, cell_id):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": source}


def insert_into_nb(nb_path, anchor_id, md_id, code_id):
    """Insert after anchor_id cell."""
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    # Remove existing cells with same ids if re-running
    nb['cells'] = [c for c in nb['cells'] if c.get('id') not in (md_id, code_id)]

    idx = next((i for i, c in enumerate(nb['cells']) if c.get('id') == anchor_id), None)
    if idx is None:
        print(f"WARNING: anchor '{anchor_id}' not found in {nb_path.name}")
        return
    insert_at = idx + 1

    nb['cells'] = nb['cells'][:insert_at] + [make_md(MD_SRC, md_id), make_code(CODE_SRC, code_id)] + nb['cells'][insert_at:]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Patched {nb_path.name}: {len(nb['cells'])} cells, inserted at {insert_at}")
    for i in range(insert_at - 1, insert_at + 3):
        c = nb['cells'][i]
        src = ''.join(c.get('source', []))
        print(f"  {i:2d} {c.get('id',''):14s} {c['cell_type']:8s} {src.split(chr(10))[0][:55]}")


# Insert after cell 3 (d388837a/d388ca151) which defines IMG_PATH and ROI_PATH
insert_into_nb(
    ROOT / 'notebooks/tube_detection_step_by_step_cam152.ipynb',
    anchor_id='d388837a',
    md_id='1bhommd152',
    code_id='1bhom_152',
)
insert_into_nb(
    ROOT / 'notebooks/tube_detection_step_by_step_cam151.ipynb',
    anchor_id='d388ca151',
    md_id='1bhommd151',
    code_id='1bhom_151',
)
