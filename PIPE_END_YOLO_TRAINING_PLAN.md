# Pipe-End YOLO Annotation And Training Plan

## Objective

Train a reliable YOLO model for one class:

```text
0 pipe_end
```

The model should detect the visible cut face/start line of each inventory pipe after homography/warp. It should not detect racks, paint marks, stickers, loose floor pipes, tools, vehicles, or pipe bodies.

## Current Rule

YOLO is not yet the production detector in the MVP.

Current intended workflow:

1. Keep the MVP/classical detector stable.
2. Use notebooks to compare YOLO detections against the current warp and tube-start logic.
3. Continue annotating and training the model.
4. Promote YOLO to production only after it passes validation across multiple capture days and both cameras.

To experiment with YOLO in the backend:

```powershell
$env:PIPE_END_YOLO_ENABLED='1'
$env:PIPE_END_YOLO_DEVICE='0'
```

To force the classical detector:

```powershell
$env:PIPE_END_YOLO_ENABLED='0'
```

## Annotation Rules

Annotate:

- Tight box around the visible cut face/start edge of each gray inventory pipe.
- Only the pipe end, not the pipe body.
- Partially occluded pipe end if at least 50 percent of the end is visible.
- Background/empty images should remain with empty labels.

Do not annotate:

- Racks, supports, structures.
- Paint marks, labels, stickers.
- Loose pipes on the ground if they are not inventory.
- Objects such as trash, tools, vehicles.
- Pipe ends smaller than roughly 8 px wide.
- A pipe cut by the image border unless at least 50 percent of the end is visible.

Consistency matters more than perfect aesthetics. Boxes should be tight and repeatable.

## Repo Layout For YOLO

Code:

```text
apps/pipe_end_annotator/annotate_app.py
pipe_end_detection/scripts/prepare_yolo_dataset.py
pipe_end_detection/scripts/train_yolo.py
pipe_end_detection/scripts/predict_unlabeled.py
src/pipe_end_yolo/inference.py
```

Local/generated data:

```text
pipe_end_detection/annotation_pool/images/
pipe_end_detection/annotation_pool/labels/
pipe_end_detection/annotation_pool/image_status.json
pipe_end_detection/dataset/
pipe_end_detection/runs/
pipe_end_detection/predictions/
pipe_end_detection/captures/
models/pipe_end_active/best.pt
```

Git policy:

- Commit code, docs, configs, and labels.
- Do not commit raw captures, warped images, generated datasets, training runs, or `.pt` weights through normal Git.
- Use Git LFS or external storage for model files and image datasets if they must be shared.

## Launch Annotation App

From repo root:

```powershell
python apps\pipe_end_annotator\annotate_app.py --port 8765 --no-browser
```

Open:

```text
http://127.0.0.1:8765/
```

Expected controls:

- Draw boxes manually.
- Save labels manually.
- Training is manual only.
- Prediction generation is manual only.
- Use `Load AI boxes` to load predictions into the current image for correction.

Do not auto-train on every saved image. Save several corrected images, then train manually.

## Continue Annotation Cycle

Recommended loop:

1. Filter to unlabeled images.
2. Annotate 10 to 20 images.
3. Save each image.
4. Click `Train AI` only after a batch is saved.
5. Click `Generate predictions`.
6. Open predicted images.
7. Click `Load AI boxes`.
8. Correct boxes: move, resize, delete false positives, add missed pipes.
9. Save corrected labels.
10. Repeat.

Track bad images:

- Use `Bad warp` for images whose warp is wrong enough that annotation would teach the model bad geometry.
- Bad warp images should be excluded from dataset preparation.

## Training On The Stronger GPU Computer

Use the stronger GPU machine for training.

Recommended setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install CUDA-enabled PyTorch if the default install does not detect the GPU. Verify:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
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

By default this training command uses only images whose YOLO label file contains at least one box. Unannotated images stay in the annotation pool for later review, but they are not treated as background examples.

Only use this when you intentionally want empty/unlabeled images as background training data:

```powershell
python pipe_end_detection\scripts\train_yolo.py --include-unlabeled-background
```

If GPU memory allows it, increase only `--batch` first. Keep `imgsz=1280` unless a validation run proves another size is better.

Useful GPU checks:

```powershell
nvidia-smi
```

YOLO should show a Python process using GPU memory during training.

## Generate Predictions For Review

After training:

```powershell
python pipe_end_detection\scripts\predict_unlabeled.py `
  --weights pipe_end_detection\runs\pipe_end\pipe_end_active\weights\best.pt `
  --imgsz 1280 `
  --conf 0.20
```

Then open the annotator and use `Load AI boxes` on images with prediction badges. Predictions are not ground truth until corrected and saved.

## Promote A Model

Promote a model only after visual validation.

Candidate source:

```text
pipe_end_detection/runs/pipe_end/pipe_end_active/weights/best.pt
```

Runtime location:

```text
models/pipe_end_active/best.pt
```

Copy manually:

```powershell
New-Item -ItemType Directory -Force models\pipe_end_active
Copy-Item pipe_end_detection\runs\pipe_end\pipe_end_active\weights\best.pt models\pipe_end_active\best.pt -Force
```

Do not commit `.pt` weights through normal Git. Use Git LFS or release artifacts.

## Notebook-First Integration

Before enabling YOLO in the MVP:

1. Add cells to `notebooks/tube_detection_step_by_step_cam151.ipynb` and cam152 notebook that run `src.pipe_end_yolo.run_pipe_end_inference` on the current `homography_warp.jpg`.
2. Plot YOLO boxes over the warp.
3. Convert YOLO boxes to `x_start_list`.
4. Compare:
   - YOLO count vs classical count.
   - Missing detections.
   - Duplicate detections.
   - Tube order bottom-to-top.
   - Distance to reference line.
5. Only then enable `PIPE_END_YOLO_ENABLED=1` for a processing test.

## Minimum Validation Before Production

Required before YOLO drives MVP measurements:

- At least 50 to 100 well-annotated images per camera side if possible.
- Include empty/background images.
- Include sunny glare, cloudy, wet, different stack fullness, and partial occlusions.
- Run validation on captures not used in training.
- Compare cam151 and cam152 matched counts.
- False positives above the stack must be corrected or filtered.
- Missing lower/upper tubes must be corrected in annotations.

Suggested acceptance for an MVP trial:

- cam151 count error within 1 tube on representative images.
- cam152 count error within 1 tube on representative images.
- No repeated double-detections on the same tube end.
- Matched count close to the known manual/classical baseline for the selected run.

## Known Risks

- The current model failed on at least one recent run with zero detections for cam151.
- The April 22 baseline is not representative of newer wide captures.
- YOLO boxes alone do not yet guarantee correct tube numbering across both cameras.
- The model can overfit if trained only on a small number of similar images.
- Warped ROI changes can invalidate old labels unless labels are remapped.

For now, treat YOLO as the path forward, but keep the classical detector available until notebook validation is solid.
