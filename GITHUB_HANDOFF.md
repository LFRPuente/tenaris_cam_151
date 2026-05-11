# GitHub Handoff Guide

This document is the practical handoff for moving the Tenaris camera MVP and pipe-end YOLO work to another computer.

Current status:

- The Sorting Table MVP is the production-facing app.
- The MVP uses the classical notebook-style detector by default.
- YOLO `pipe_end` code is present, but it is experimental and disabled by default.
- The next integration step is notebook validation before enabling YOLO in production processing.

## Repository Scope

This repo should contain:

- MVP source code.
- Clean backend pipeline source code.
- Notebooks used for camera-specific validation.
- ROI/default calibration files.
- YOLO annotator app code.
- YOLO dataset preparation, training, and prediction scripts.
- YOLO labels and small metadata when practical.
- Documentation describing current assumptions and next steps.

This repo should not contain through normal Git:

- Camera credentials.
- Raw capture history.
- Large annotation images.
- Generated YOLO datasets.
- Training runs.
- Model weights such as `.pt` files.
- Generated MVP artifacts.

Use Git LFS or external storage if image datasets and model weights must live in the same GitHub project.

## Important Files

Read these first on a new machine:

- `README.md`
- `MVP_CONTEXT.md`
- `YOLO_MVP_INTEGRATION_PLAN.md`
- `PIPE_END_YOLO_TRAINING_PLAN.md`
- `pipe_end_detection/ANNOTATION_GUIDE.md`

Main runtime files:

- `run_sorting_table_mvp.py`
- `run_sorting_table_mvp_bg.py`
- `process_tube_pair.py`
- `src/tenaris_tube_pipeline/`
- `src/cam151_ref_detection/sorting_table_mvp_proc.py`
- `src/cam151_ref_detection/capture_history.py`

YOLO files:

- `apps/pipe_end_annotator/annotate_app.py`
- `pipe_end_detection/capture_pipe_end_dataset.py`
- `pipe_end_detection/scripts/prepare_yolo_dataset.py`
- `pipe_end_detection/scripts/train_yolo.py`
- `pipe_end_detection/scripts/predict_unlabeled.py`
- `src/pipe_end_yolo/inference.py`

## Clone And Setup

Clone:

```powershell
git clone https://github.com/LFRPuente/tenaris_cam_151.git
cd tenaris_cam_151
```

Create environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify imports:

```powershell
python -m compileall src apps pipe_end_detection process_tube_pair.py run_sorting_table_mvp.py run_sorting_table_mvp_bg.py
```

## Camera Configuration

Camera download uses the shared `config.json` from the local workspace.

Do not commit real credentials.

On a new machine, create the local camera config expected by the capture scripts. Confirm the config includes the same camera parameters used for wide captures, especially:

- cam151 endpoint and credentials;
- cam152 endpoint and credentials;
- zoom `0` for current YOLO capture strategy;
- intended pan/tilt defaults;
- timestamp overlays disabled for dataset images;
- camera overlays disabled for dataset images.

Start from:

```powershell
Copy-Item config.example.json config.json
```

Then edit `config.json` locally. Do not commit the real `config.json` if it contains credentials.

## Runtime Defaults

Classical detector is default:

```powershell
$env:PIPE_END_YOLO_ENABLED='0'
```

Experimental YOLO processing:

```powershell
$env:PIPE_END_YOLO_ENABLED='1'
$env:PIPE_END_YOLO_DEVICE='0'
$env:PIPE_END_YOLO_MODEL='models\pipe_end_active\best.pt'
```

Do not enable YOLO for the production MVP until notebook validation is acceptable.

## Run The MVP

Foreground:

```powershell
python run_sorting_table_mvp.py --port 58597 --no-browser
```

Background:

```powershell
python run_sorting_table_mvp_bg.py --port 58597 --url-file artifacts\sorting_table_mvp\live_url_current.txt
```

Open:

```text
http://127.0.0.1:58597/
```

History view:

```text
http://127.0.0.1:58597/history
```

## Process The April 22 Baseline

The April 22 pair is the frozen baseline for the MVP.

```powershell
python process_tube_pair.py `
  --cam151-image test_images\cam_151_202604022.jpeg `
  --cam152-image test_images\cam_152_202604022.jpeg `
  --output-stem backend_current_pair
```

Expected detector source unless explicitly changed:

```text
notebook_style_sobel_x_roi
```

## Capture YOLO Images

Single capture:

```powershell
python pipe_end_detection\capture_pipe_end_dataset.py
```

Loop every 90 minutes:

```powershell
python pipe_end_detection\capture_pipe_end_dataset.py --loop --interval-minutes 90
```

Outputs are ignored by Git:

```text
pipe_end_detection/captures/
pipe_end_detection/annotation_pool/images/
```

If these images must move to another machine, copy them separately or configure Git LFS first.

## Annotation App

Run:

```powershell
python apps\pipe_end_annotator\annotate_app.py --port 8765 --no-browser
```

Open:

```text
http://127.0.0.1:8765/
```

Annotation rule:

- Draw one tight box around each visible pipe end.
- Do not draw around the full pipe body.
- Save corrected labels manually.
- Do not train after every image.
- Train after saving a batch of corrected annotations.

## Training On The Stronger GPU

Use the stronger GPU computer for training.

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nvidia-smi
```

Train:

```powershell
python pipe_end_detection\scripts\train_yolo.py `
  --model yolo11n.pt `
  --imgsz 1280 `
  --epochs 100 `
  --batch 8 `
  --name pipe_end_active
```

Default behavior:

- images with box labels are used for training;
- unannotated images remain available in the annotator;
- unannotated images are not treated as background unless `--include-unlabeled-background` is passed explicitly.

If GPU memory allows it, increase `--batch` first. Do not reduce `--epochs` or `--imgsz` unless a controlled validation run shows better results.

Generate predictions:

```powershell
python pipe_end_detection\scripts\predict_unlabeled.py `
  --weights pipe_end_detection\runs\pipe_end\pipe_end_active\weights\best.pt `
  --imgsz 1280 `
  --conf 0.20
```

Review predictions in the annotator using `Load AI boxes`, correct them, and save them as labels.

## Promote A YOLO Model

Only promote after visual validation in notebooks.

```powershell
New-Item -ItemType Directory -Force models\pipe_end_active
Copy-Item pipe_end_detection\runs\pipe_end\pipe_end_active\weights\best.pt models\pipe_end_active\best.pt -Force
```

Model weights are ignored by normal Git. Use Git LFS or release artifacts if the model must be shared through GitHub.

## Notebook-First YOLO Integration

Before YOLO drives the MVP table:

1. Open `notebooks/tube_detection_step_by_step_cam151.ipynb`.
2. Run the classical pipeline cells for the selected history image.
3. Add/run YOLO inference on the same `homography_warp.jpg`.
4. Plot YOLO boxes on the warp.
5. Convert YOLO boxes to ordered tube-start positions.
6. Compare count, duplicates, missed tubes, ordering, and reference-line distances.
7. Repeat the same validation for `notebooks/tube_detection_step_by_step_cam152.ipynb`.

Only after both cameras pass representative history images should `PIPE_END_YOLO_ENABLED=1` be used for MVP processing tests.

## Data Transfer Checklist

If moving to a new computer without Git LFS, manually copy:

- `pipe_end_detection/annotation_pool/images/`
- `pipe_end_detection/annotation_pool/labels/`
- `pipe_end_detection/annotation_pool/image_status.json`
- `pipe_end_detection/captures/`
- `models/pipe_end_active/best.pt`
- any candidate `pipe_end_detection/runs/.../weights/best.pt`
- local `config.json` with camera credentials, handled outside Git

Then run:

```powershell
python -m compileall src apps pipe_end_detection process_tube_pair.py
```

## Current Technical Debt

- YOLO failed on at least one recent cam151 warp with zero detections.
- The current YOLO model is not yet representative enough for production MVP processing.
- The notebooks need explicit YOLO comparison cells.
- Image datasets and model weights need a Git LFS or external storage decision.
- `config.json` credential handling should be formalized with a safe example file.

## Recommended Next Work Order

1. Push code/docs only to GitHub.
2. Decide whether to use Git LFS for images and `.pt` files.
3. Transfer existing annotation images/labels to the GPU computer.
4. Continue annotation in batches of 10 to 20 images.
5. Train manually after each annotation batch.
6. Generate predictions only when requested.
7. Correct predictions and save them as labels.
8. Validate YOLO in notebooks against classical output.
9. Add MVP overlay visualization for YOLO boxes.
10. Enable YOLO in backend processing only after validation passes.
