from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from .roi_store import load_rois, save_rois


@dataclass
class RoiItem:
    id: str
    xyxy: tuple[int, int, int, int]


class RoiPickerApp(tk.Tk):
    def __init__(self, image_path: str | Path, save_path: str | Path, initial: list[dict] | None = None):
        super().__init__()
        self.title(f"ROI Picker - {Path(image_path).name}")
        self.geometry("1500x980")

        self.image_path = Path(image_path)
        self.save_path = Path(save_path)
        self.initial = initial or []

        self.original_image = Image.open(self.image_path).convert("RGB")
        self.original_width, self.original_height = self.original_image.size

        max_w, max_h = 1350, 820
        self.scale = min(max_w / self.original_width, max_h / self.original_height, 1.0)
        self.display_width = max(1, int(round(self.original_width * self.scale)))
        self.display_height = max(1, int(round(self.original_height * self.scale)))
        self.display_image = self.original_image.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)

        self.rois: list[RoiItem] = []
        self.selected_index: int | None = None
        self.start_xy: tuple[int, int] | None = None
        self.preview_rect_id: int | None = None
        self.canvas_rect_ids: list[int] = []

        self._build_ui()
        self._load_initial_rois()
        self._redraw()

    def _build_ui(self) -> None:
        help_text = (
            "Left drag: add ROI | Right click: select ROI | Delete: remove selected | "
            "C: clear | S/Ctrl+S: save | Esc: close"
        )
        ttk.Label(self, text=help_text).pack(anchor="w", padx=10, pady=(8, 4))

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=(0, 8))

        ttk.Button(controls, text="Save", command=self._save).pack(side="left")
        ttk.Button(controls, text="Delete Selected", command=self._delete_selected).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Clear All", command=self._clear_all).pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="No ROI selected.")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=(14, 0))

        self.canvas = tk.Canvas(self, width=self.display_width, height=self.display_height, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        self.canvas.bind("<ButtonPress-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self.bind("<Delete>", lambda _e: self._delete_selected())
        self.bind("<BackSpace>", lambda _e: self._delete_selected())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.bind("<KeyPress-c>", lambda _e: self._clear_all())
        self.bind("<KeyPress-C>", lambda _e: self._clear_all())
        self.bind("<KeyPress-s>", lambda _e: self._save())
        self.bind("<Control-s>", lambda _e: self._save())

    def _load_initial_rois(self) -> None:
        for index, roi in enumerate(self.initial, start=1):
            xyxy = tuple(int(v) for v in roi["xyxy"])
            self.rois.append(RoiItem(id=f"roi_{index:02d}", xyxy=xyxy))

    def _canvas_to_image(self, x: float, y: float) -> tuple[int, int]:
        xi = int(round(max(0.0, min(float(x) / self.scale, self.original_width - 1))))
        yi = int(round(max(0.0, min(float(y) / self.scale, self.original_height - 1))))
        return xi, yi

    def _image_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))

    def _normalize_rect(self, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        xa, xb = sorted((x1, x2))
        ya, yb = sorted((y1, y2))
        xa = max(0, min(xa, self.original_width - 1))
        xb = max(xa + 1, min(xb, self.original_width))
        ya = max(0, min(ya, self.original_height - 1))
        yb = max(ya + 1, min(yb, self.original_height))
        return xa, ya, xb, yb

    def _redraw(self) -> None:
        for item_id in self.canvas_rect_ids:
            self.canvas.delete(item_id)
        self.canvas_rect_ids.clear()

        for index, roi in enumerate(self.rois, start=1):
            x1, y1, x2, y2 = roi.xyxy
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)
            color = "#00ff00" if self.selected_index == (index - 1) else "#ffd400"
            rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2)
            text_id = self.canvas.create_text(cx1 + 8, max(12, cy1 - 10), text=f"ROI {index}", anchor="nw", fill=color, font=("Segoe UI", 11, "bold"))
            self.canvas_rect_ids.extend([rect_id, text_id])

        if self.selected_index is None:
            self.status_var.set(f"ROIs: {len(self.rois)}")
        else:
            roi = self.rois[self.selected_index]
            self.status_var.set(f"Selected ROI {self.selected_index + 1}: {roi.xyxy}")

    def _on_left_down(self, event) -> None:
        self.start_xy = self._canvas_to_image(event.x, event.y)
        if self.preview_rect_id is not None:
            self.canvas.delete(self.preview_rect_id)
            self.preview_rect_id = None

    def _on_left_drag(self, event) -> None:
        if self.start_xy is None:
            return
        x1, y1 = self.start_xy
        x2, y2 = self._canvas_to_image(event.x, event.y)
        ix1, iy1, ix2, iy2 = self._normalize_rect(x1, y1, x2, y2)
        cx1, cy1 = self._image_to_canvas(ix1, iy1)
        cx2, cy2 = self._image_to_canvas(ix2, iy2)
        if self.preview_rect_id is not None:
            self.canvas.delete(self.preview_rect_id)
        self.preview_rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00ffff", width=2, dash=(6, 4))

    def _on_left_up(self, event) -> None:
        if self.start_xy is None:
            return
        x1, y1 = self.start_xy
        x2, y2 = self._canvas_to_image(event.x, event.y)
        self.start_xy = None
        if self.preview_rect_id is not None:
            self.canvas.delete(self.preview_rect_id)
            self.preview_rect_id = None

        ix1, iy1, ix2, iy2 = self._normalize_rect(x1, y1, x2, y2)
        if (ix2 - ix1) < 6 or (iy2 - iy1) < 6:
            return
        roi_id = f"roi_{len(self.rois) + 1:02d}"
        self.rois.append(RoiItem(id=roi_id, xyxy=(ix1, iy1, ix2, iy2)))
        self.selected_index = len(self.rois) - 1
        self._redraw()

    def _on_right_click(self, event) -> None:
        ix, iy = self._canvas_to_image(event.x, event.y)
        self.selected_index = None
        for index, roi in enumerate(self.rois):
            x1, y1, x2, y2 = roi.xyxy
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                self.selected_index = index
                break
        self._redraw()

    def _delete_selected(self) -> None:
        if self.selected_index is None:
            return
        del self.rois[self.selected_index]
        self.selected_index = None
        self._redraw()

    def _clear_all(self) -> None:
        self.rois.clear()
        self.selected_index = None
        self._redraw()

    def _save(self) -> None:
        rois_payload = [{"xyxy": list(roi.xyxy)} for roi in self.rois]
        payload = save_rois(self.save_path, self.image_path, rois_payload)
        self.status_var.set(f"Saved {len(payload['rois'])} ROIs to {self.save_path.name}")
        messagebox.showinfo("ROI Picker", f"Saved {len(payload['rois'])} ROIs.\n\nJSON: {self.save_path}")


def pick_rois_tk(image_path: str | Path, save_path: str | Path, initial: list[dict] | None = None) -> list[dict]:
    app = RoiPickerApp(image_path, save_path, initial=initial)
    app.mainloop()
    return list(load_rois(save_path).get("rois", []))
