from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .capture_history import default_roi_paths, list_capture_run_manifests, load_latest_capture_run_manifest, repo_root


@dataclass(frozen=True)
class NotebookHistorySelection:
    side: str
    selection_id: str
    source_type: str
    label: str
    image_path: Path
    roi_path: Path
    match_json_path: Path | None
    cam151_image_path: Path | None
    cam152_image_path: Path | None
    captured_at: str
    summary: dict[str, int]
    metadata: dict[str, Any]


def _normalize_side(side: Any) -> str:
    raw = str(side or "").strip().lower()
    if raw in {"151", "cam151", "cam_151", "left", "izq", "izquierda"}:
        return "151"
    if raw in {"152", "cam152", "cam_152", "right", "der", "derecha"}:
        return "152"
    raise ValueError(f"Lado no soportado: {side!r}")


def _resolve_existing_path(raw_value: Any) -> Path | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    root = repo_root()
    candidates = [
        root / text,
        root / "test_images" / Path(text).name,
        root / "manual_rois" / Path(text).name,
        root / "artifacts" / "tube_matching" / Path(text).name,
        root / "artifacts" / "tube_matcher_inputs" / Path(text).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON invalido en {path}")
    return payload


def _default_artifact_match_dir() -> Path:
    return repo_root() / "artifacts" / "tube_matching"


def _pipe_end_capture_runs_dir() -> Path:
    return repo_root() / "pipe_end_detection" / "captures" / "runs"


def _resolve_capture_manifest_path(path_like: Any) -> Path | None:
    raw = str(path_like or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(repo_root() / candidate)
    seen: set[str] = set()
    for entry in candidates:
        key = str(entry)
        if key in seen:
            continue
        seen.add(key)
        if entry.is_dir():
            manifest_path = entry / "manifest.json"
            if manifest_path.exists():
                return manifest_path
        if entry.is_file() and entry.name.lower() == "manifest.json":
            return entry
    return None


def _format_summary(summary: dict[str, Any]) -> str:
    matched = int(summary.get("matched") or 0)
    left_only = int(summary.get("left_only") or 0)
    right_only = int(summary.get("right_only") or 0)
    return f"matched={matched} left_only={left_only} right_only={right_only}"


def _format_display_time(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return "-"
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _build_capture_selection(manifest: dict[str, Any], side: str) -> NotebookHistorySelection | None:
    side_key = _normalize_side(side)
    camera_key = f"cam{side_key}"
    camera_info = dict((manifest.get("cameras") or {}).get(camera_key) or {})
    processing = dict(manifest.get("processing") or {})
    image_path = _resolve_existing_path(camera_info.get("image_path"))
    if image_path is None or not image_path.exists():
        return None
    roi_path = _resolve_existing_path((manifest.get("roi_paths") or {}).get(camera_key)) or default_roi_paths()[side_key]
    match_json_path = _resolve_existing_path(processing.get("match_latest_json_path") or processing.get("match_json_path"))
    cam151_path = _resolve_existing_path(((manifest.get("cameras") or {}).get("cam151") or {}).get("image_path"))
    cam152_path = _resolve_existing_path(((manifest.get("cameras") or {}).get("cam152") or {}).get("image_path"))
    summary = {
        "matched": int(((processing.get("summary") or {}).get("matched")) or 0),
        "left_only": int(((processing.get("summary") or {}).get("left_only")) or 0),
        "right_only": int(((processing.get("summary") or {}).get("right_only")) or 0),
    }
    run_id = str(manifest.get("run_id") or "").strip()
    captured_at = str(manifest.get("captured_at") or "")
    image_name = str(camera_info.get("image_name") or image_path.name)
    label = f"[captura] {run_id} | {_format_display_time(captured_at)} | {_format_summary(summary)} | {image_name}"
    return NotebookHistorySelection(
        side=side_key,
        selection_id=run_id,
        source_type="capture_run",
        label=label,
        image_path=image_path,
        roi_path=roi_path,
        match_json_path=match_json_path,
        cam151_image_path=cam151_path,
        cam152_image_path=cam152_path,
        captured_at=captured_at,
        summary=summary,
        metadata={"manifest": manifest},
    )


def _build_artifact_selection(match_path: Path, side: str) -> NotebookHistorySelection | None:
    side_key = _normalize_side(side)
    payload = _read_json(match_path)
    if match_path.name.lower() == "tube_match_latest.json":
        return None
    inputs = dict(payload.get("inputs") or {})
    side_info = dict(inputs.get(f"cam{side_key}") or {})
    image_path = _resolve_existing_path(side_info.get("image_path"))
    if image_path is None or not image_path.exists():
        return None
    roi_path = _resolve_existing_path(side_info.get("roi_path")) or default_roi_paths()[side_key]
    cam151_path = _resolve_existing_path(((inputs.get("cam151") or {}).get("image_path")) or "")
    cam152_path = _resolve_existing_path(((inputs.get("cam152") or {}).get("image_path")) or "")
    summary = dict(payload.get("summary") or {})
    summary_norm = {
        "matched": int(summary.get("matched") or 0),
        "left_only": int(summary.get("left_only") or 0),
        "right_only": int(summary.get("right_only") or 0),
    }
    captured_at = str(payload.get("generated_at") or "")
    label = f"[artefacto] {match_path.name} | {_format_display_time(captured_at)} | {_format_summary(summary_norm)} | {image_path.name}"
    return NotebookHistorySelection(
        side=side_key,
        selection_id=f"artifact:{match_path.name}",
        source_type="artifact",
        label=label,
        image_path=image_path,
        roi_path=roi_path,
        match_json_path=match_path,
        cam151_image_path=cam151_path,
        cam152_image_path=cam152_path,
        captured_at=captured_at,
        summary=summary_norm,
        metadata={"match_payload": payload},
    )


def list_notebook_history_entries(side: Any, *, include_latest_capture: bool = False) -> list[NotebookHistorySelection]:
    side_key = _normalize_side(side)
    entries: list[NotebookHistorySelection] = []
    seen_ids: set[str] = set()

    for manifest in list_capture_run_manifests():
        selection = _build_capture_selection(manifest, side_key)
        if selection is None:
            continue
        seen_ids.add(selection.selection_id)
        entries.append(selection)

    if include_latest_capture:
        latest_manifest = load_latest_capture_run_manifest()
        if latest_manifest is not None:
            latest_selection = _build_capture_selection(latest_manifest, side_key)
            if latest_selection is not None and latest_selection.selection_id not in seen_ids:
                entries.append(latest_selection)
                seen_ids.add(latest_selection.selection_id)

    pipe_end_runs_dir = _pipe_end_capture_runs_dir()
    if pipe_end_runs_dir.exists():
        for manifest_path in sorted(pipe_end_runs_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            selection = _build_capture_selection(_read_json(manifest_path), side_key)
            if selection is None or selection.selection_id in seen_ids:
                continue
            seen_ids.add(selection.selection_id)
            entries.append(selection)

    for match_path in sorted(_default_artifact_match_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        selection = _build_artifact_selection(match_path, side_key)
        if selection is None:
            continue
        entries.append(selection)

    entries.sort(key=lambda item: str(item.captured_at or item.selection_id), reverse=True)
    return entries


def resolve_notebook_history_selection(side: Any, selection_id: str) -> NotebookHistorySelection:
    side_key = _normalize_side(side)
    wanted = str(selection_id or "").strip()
    if not wanted:
        raise ValueError("selection_id vacio.")

    if wanted.lower() == "latest":
        latest_manifest = load_latest_capture_run_manifest()
        if latest_manifest is None:
            raise FileNotFoundError("No hay latest capture registrado.")
        selection = _build_capture_selection(latest_manifest, side_key)
        if selection is None:
            raise FileNotFoundError("El latest capture no tiene imagen utilizable para notebook.")
        return selection

    if wanted.startswith("artifact:"):
        wanted = wanted.split(":", 1)[1]

    direct_manifest_path = _resolve_capture_manifest_path(wanted)
    if direct_manifest_path is not None:
        selection = _build_capture_selection(_read_json(direct_manifest_path), side_key)
        if selection is None:
            raise FileNotFoundError(f"La corrida {wanted!r} no tiene imagen utilizable para notebook.")
        return selection

    manifest_path = repo_root() / "artifacts" / "capture_history" / wanted / "manifest.json"
    if manifest_path.exists():
        selection = _build_capture_selection(_read_json(manifest_path), side_key)
        if selection is None:
            raise FileNotFoundError(f"La corrida {wanted!r} no tiene imagen utilizable para notebook.")
        return selection

    manifest_path = _pipe_end_capture_runs_dir() / wanted / "manifest.json"
    if manifest_path.exists():
        selection = _build_capture_selection(_read_json(manifest_path), side_key)
        if selection is None:
            raise FileNotFoundError(f"La corrida {wanted!r} no tiene imagen utilizable para notebook.")
        return selection

    match_dir = _default_artifact_match_dir()
    artifact_candidates = [
        match_dir / wanted,
        match_dir / f"{wanted}.json" if not wanted.lower().endswith(".json") else None,
    ]
    for candidate in artifact_candidates:
        if candidate is None or not candidate.exists():
            continue
        selection = _build_artifact_selection(candidate, side_key)
        if selection is None:
            break
        return selection

    available = [entry.selection_id for entry in list_notebook_history_entries(side_key)]
    raise FileNotFoundError(f"No se encontro selection_id={selection_id!r}. Disponibles: {available[:12]}")


def pick_notebook_history_selection(side: Any, *, title: str | None = None) -> NotebookHistorySelection | None:
    entries = list_notebook_history_entries(side)
    if not entries:
        return None

    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - local GUI dependency
        raise RuntimeError("Tkinter no esta disponible en este entorno.") from exc

    chosen: dict[str, Any] = {"selection": None}
    filtered_entries = list(entries)

    root = tk.Tk()
    root.title(title or f"Seleccionar historial notebook cam{_normalize_side(side)}")
    root.geometry("1260x540")
    root.minsize(980, 420)

    top = tk.Frame(root)
    top.pack(fill="x", padx=12, pady=(12, 8))
    tk.Label(top, text="Filtro:").pack(side="left")
    filter_var = tk.StringVar()
    filter_entry = tk.Entry(top, textvariable=filter_var)
    filter_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))

    info_var = tk.StringVar(value=f"{len(entries)} opcion(es)")
    tk.Label(root, textvariable=info_var, anchor="w").pack(fill="x", padx=12)

    listbox = tk.Listbox(root, width=180, height=20)
    listbox.pack(fill="both", expand=True, padx=12, pady=(6, 10))

    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=12, pady=(0, 12))

    def refresh_list(*_args: Any) -> None:
        text = str(filter_var.get() or "").strip().lower()
        filtered_entries.clear()
        listbox.delete(0, tk.END)
        for entry in entries:
            haystack = " ".join(
                [
                    entry.selection_id,
                    entry.label,
                    entry.image_path.name,
                    entry.roi_path.name,
                ]
            ).lower()
            if text and text not in haystack:
                continue
            filtered_entries.append(entry)
            listbox.insert(tk.END, entry.label)
        info_var.set(f"{len(filtered_entries)} opcion(es) visibles de {len(entries)} total(es)")
        if filtered_entries:
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(0)
            listbox.activate(0)

    def confirm_selection(*_args: Any) -> None:
        current = listbox.curselection()
        if not current:
            return
        chosen["selection"] = filtered_entries[int(current[0])]
        root.destroy()

    def cancel_selection(*_args: Any) -> None:
        root.destroy()

    tk.Button(buttons, text="Usar seleccion", command=confirm_selection).pack(side="left")
    tk.Button(buttons, text="Cancelar", command=cancel_selection).pack(side="left", padx=(8, 0))

    filter_var.trace_add("write", refresh_list)
    listbox.bind("<Double-Button-1>", confirm_selection)
    listbox.bind("<Return>", confirm_selection)
    root.bind("<Escape>", cancel_selection)

    refresh_list()
    filter_entry.focus_set()
    root.mainloop()
    return chosen["selection"]


def pick_notebook_history_folder_selection(side: Any, *, title: str | None = None) -> NotebookHistorySelection | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - local GUI dependency
        raise RuntimeError("Tkinter no esta disponible en este entorno.") from exc

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    chosen_dir = filedialog.askdirectory(
        title=title or f"Seleccionar folder de historial cam{_normalize_side(side)}",
        initialdir=str(repo_root() / "artifacts" / "capture_history"),
        mustexist=True,
        parent=root,
    )
    root.destroy()
    if not chosen_dir:
        return None
    return resolve_notebook_history_selection(side, chosen_dir)
