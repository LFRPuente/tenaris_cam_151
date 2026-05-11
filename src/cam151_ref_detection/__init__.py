"""Bootstrap detection package for cam151."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "Cam151Config": (".config", "Cam151Config"),
    "run_cam151_bootstrap": (".pipeline", "run_cam151_bootstrap"),
    "pick_rois_tk": (".roi_picker", "pick_rois_tk"),
    "load_rois": (".roi_store", "load_rois"),
    "save_rois": (".roi_store", "save_rois"),
    "NotebookHistorySelection": (".notebook_history", "NotebookHistorySelection"),
    "list_notebook_history_entries": (".notebook_history", "list_notebook_history_entries"),
    "pick_notebook_history_folder_selection": (".notebook_history", "pick_notebook_history_folder_selection"),
    "pick_notebook_history_selection": (".notebook_history", "pick_notebook_history_selection"),
    "resolve_notebook_history_selection": (".notebook_history", "resolve_notebook_history_selection"),
    "build_matcher_input_dataset": (".tube_matcher_proc", "build_matcher_input_dataset"),
    "export_tube_measurements": (".tube_matcher_proc", "export_tube_measurements"),
    "find_latest_measurement_export": (".tube_matcher_proc", "find_latest_measurement_export"),
    "run_tube_matcher": (".tube_matcher_proc", "run_tube_matcher"),
    "run_manual_tube_measure_app": (".manual_tube_measure_proc", "run_manual_tube_measure_app"),
    "run_scale_line_picker": (".scale_line_picker_proc", "run_scale_line_picker"),
    "start_sorting_table_mvp_server": (".sorting_table_mvp_proc", "start_sorting_table_mvp_server"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
