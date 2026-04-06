# Cam151 Automatic Ref Detection

Clean bootstrap project for building a dedicated automatic detection pipeline for `cam151`.

This repo is intentionally smaller than the older `automatic_ref_detection` workspace:

- no legacy notebook sprawl
- no cross-camera assumptions
- no SAM dependency in the first bootstrap step

Current goal:

1. process new `cam151` wide images
2. detect stable geometric/green structures
3. generate debug artifacts and a machine-readable summary
4. use that output to design the real `cam151` detection pipeline

## Layout

- `PROJECT_CONTEXT.md`: current project context and next steps
- `test_images/`: local `cam151` images copied into the repo
- `src/cam151_ref_detection/`: minimal Python package
- `run_cam151_detection.py`: CLI entrypoint
- `run_roi_web_server.py`: local web server for multi-ROI selection
- `web_roi_picker/`: HTML/CSS/JS ROI picker
- `artifacts/`: generated outputs
- `manual_rois/`: saved ROI JSON/crops (generated locally, gitignored)

## Image set

Images live inside `test_images/` and the notebook lets you choose by index.

## How to run

```powershell
python run_cam151_detection.py ".\test_images\cam151_wide_20260323_120028.jpg"
```

## ROI web app

```powershell
python run_roi_web_server.py
```

Then open:

- `http://127.0.0.1:8765/web_roi_picker/`

The app supports:

- multiple ROIs per image
- selection/deletion
- saving ROI JSON to `manual_rois/`
- exporting numbered crops and an overview image
