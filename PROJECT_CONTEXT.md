# Project Context - Cam151 Bootstrap

## Why this repo exists

The older `automatic_ref_detection` project has useful logic, but the current working assumption is that the mature flow is mostly oriented to `cam152`.

`cam151` needs its own cleaner bootstrap because:

- the geometry is different
- the framing is different
- the operational camera behavior is different
- we want to work from new production images, not only legacy notebook examples

## Current image source

Initial processing targets copied into this repo under `test_images/`.

Current set:

- `cam151_wide_20260323_100046.jpg`
- `cam151_wide_20260323_102005.jpg`
- `cam151_wide_20260323_103026.jpg`
- `cam151_wide_20260323_120028.jpg`
- `cam151_wide_20260323_140058.jpg`

## Current bootstrap scope

This repo does not try to solve final mark detection yet.

For now it does:

1. load a `cam151` image
2. detect green structural elements using HSV + connected components
3. define a bottom work area
4. compute edges/line candidates in that area
5. export visual artifacts and `summary.json`
6. let the user define multiple ROIs through a local web app and save them as numbered crops/json

## Intended next stage

After we inspect bootstrap outputs, we should decide one of these directions:

1. dedicated `cam151` ROI/warp pipeline
2. tube strip extraction for the bottom bundle
3. mark detection on a normalized warp
4. hybrid pipeline with manual seed points + automatic refinement

## Design rules

- keep code modular and small
- keep config isolated in one file
- save every important intermediate artifact
- avoid inheriting notebook-only assumptions from the old repo

---

## Current Working Context - 2026-05-18

This repository is now being used as the active workspace for the pipe-end and tube-bundle YOLO annotation/training loop, plus the cam151/cam152 matching work. The old bootstrap context above is historical; the operational state below is the current handoff source.

### Active workspace

- Repo root: `C:\Users\luis_\OneDrive\Desktop\tenaris_cam_151`
- Annotation app: `apps/pipe_end_annotator/annotate_app.py`
- Local app URL: `http://127.0.0.1:8765/`
- Current app launch uses GPU training: `--train-device 0`
- CUDA environment verified in `.venv`:
  - `torch 2.11.0+cu128`
  - GPU: `NVIDIA GeForce RTX 5080`
  - `torch.cuda.is_available()` is `True`

### Annotation tasks in the app

The annotator now has four separate tasks:

- `pipe_end` / `Pipe ends: cam151 model`
  - Class: `pipe_end`
  - Labels: `pipe_end_detection/annotation_pool/labels/cam151`
  - Active model: `models/pipe_end_active/best.pt`
  - Allows explicit negative labels through the `Negative image` button.

- `pipe_end_cam152` / `Pipe ends: cam152 model`
  - Class: `pipe_end`
  - Labels share the same root but are filtered to `cam152`.
  - Active model: `models/pipe_end_cam152_active/best.pt`
  - Allows explicit negative labels through the `Negative image` button.
  - It must not fall back to the cam151 model. First training should initialize from the configured base model, currently `yolo11m.pt`.

- `tube_bundle` / `Tube bundle: one box per image`
  - Class: `tube_bundle`
  - Labels: `tube_bundle_detection/annotation_pool/labels`
  - Active model: `models/tube_bundle_active/best.pt`
  - Uses one box per image.
  - The first bundle training included all unlabeled images as negatives; the current experiment limits negatives intentionally.

- `tube_bundle_edge` / `Tube bundle edge: pipe-end strip`
  - Class: `tube_bundle_edge`
  - Labels: `tube_bundle_edge_detection/annotation_pool/labels`
  - Active model: `models/tube_bundle_edge_active/best.pt`
  - Uses one box per image.
  - The box should cover the narrow visible pipe-end strip/edge, not the full tube bundle.
  - Allows explicit negative labels through the `Negative image` button.
  - Unannotated images are not treated as negatives, because this task is only useful when the visible edge is intentionally reviewed.

### Important annotator code changes

`apps/pipe_end_annotator/annotate_app.py` now includes:

- task-specific model paths for cam151 pipe ends, cam152 pipe ends, and tube bundle;
- task-specific model path for `tube_bundle_edge`;
- `include_unlabeled_negatives` for bundle training;
- `allow_negative_labels` for cam151/cam152 pipe-end training and the `tube_bundle_edge` task;
- status storage for explicit negatives under `negative_labels`;
- `Negative image` / `Clear negative` UI support for cam151/cam152 pipe-end tasks and `tube_bundle_edge`;
- filtered task lists, badges, and label save/load behavior that preserves explicit negative annotations;
- single-image prediction support per selected task, so `Run model on this image` only runs the active task/model.
- cam152 fallback to cam151 was removed from prediction and initial training logic.
- selected boxes now show resize handles on corners/sides; drag a handle to resize instead of redrawing the box.

`pipe_end_detection/README_ANNOTATOR.md` was updated to document negative annotations.

### Model inventory

Active model paths:

- cam151 pipe ends: `models/pipe_end_active/best.pt`
- cam152 pipe ends: `models/pipe_end_cam152_active/best.pt`
- tube bundle: `models/tube_bundle_active/best.pt`
- tube bundle edge strip: `models/tube_bundle_edge_active/best.pt`

The current `models/tube_bundle_active/best.pt` was restored to the previous good `yolo11n` bundle model after the first `yolo11m` bundle experiment performed worse.

### Tube bundle training history

First completed bundle model:

- Base model: `yolo11n.pt`
- Dataset: 250 total images
- Split:
  - train: 200 total, 135 positive, 65 negative
  - val: 50 total, 36 positive, 14 negative
- Final validation:
  - precision: about `0.914`
  - recall: about `0.885`
  - mAP50: about `0.940`
  - mAP50-95: about `0.637`
- Active weight after that run: `models/tube_bundle_active/best.pt`

First `yolo11m` bundle experiment:

- Base model: `yolo11m.pt`
- Dataset: same 250 images with many negatives
- Config: `imgsz=1536`, `epochs=120`, `batch=2`, `device=0`
- It was stopped around epoch 96 because it clearly underperformed:
  - best mAP50-95 seen was about `0.260`
  - last mAP50-95 before stopping was about `0.235`
- The previous good `yolo11n` bundle model was restored afterward.

Current bundle experiment:

- Purpose: test the hypothesis that too many negatives hurt the bundle model.
- Base model: `yolo11m.pt`
- Dataset root: `tube_bundle_detection/active_training_limited_negatives`
- Run root: `tube_bundle_detection/runs/tube_bundle_yolo11m_limited_negatives/latest`
- Config: `imgsz=1536`, `epochs=120`, `batch=2`, `device=0`
- Dataset:
  - all available positives: 171
  - candidate negatives: 79
  - train: 142 total, 137 positive, 5 negative, 137 instances
  - val: 37 total, 34 positive, 3 negative, 34 instances
- Logs:
  - `.logs/tube_bundle_yolo11m_limited_negatives.out.log`
  - `.logs/tube_bundle_yolo11m_limited_negatives.err.log`
  - `.logs/tube_bundle_yolo11m_limited_negatives.copy.log`
  - `.logs/tube_bundle_yolo11m_limited_negatives.pid`
- When it finishes, a watcher copies its `best.pt` to `models/tube_bundle_active/best.pt` and backs up the previous active model first.

### Queued pipe-end trainings

The previous cam151 partial `yolo11m` training and old cam152 queue were intentionally stopped before restarting the limited-negative bundle experiment.

A new clean queue is active:

- Queue PID is stored in `.logs/pipe_end_yolo11m_after_bundle_limited.queue.pid`
- Queue log: `.logs/pipe_end_yolo11m_after_bundle_limited.queue.log`
- It waits for the current bundle limited-negative training PID.
- Then it trains cam151 pipe ends.
- Then it trains cam152 pipe ends, initializing from the active cam151 model if available.

Queued cam151 training:

- Dataset root: `pipe_end_detection/active_training/dataset`
- Data YAML: `pipe_end_detection/active_training/data_active.yaml`
- Run root: `pipe_end_detection/runs/pipe_end_cam151_yolo11m_after_bundle_limited/latest`
- Active output: `models/pipe_end_active/best.pt`
- Config: `yolo11m.pt`, `imgsz=1536`, `epochs=120`, `batch=2`, `device=0`
- Dataset:
  - train: 75 images, 75 positive, 0 negative, 2388 pipe-end instances
  - val: 19 images, 19 positive, 0 negative, 637 pipe-end instances
- Logs:
  - `.logs/pipe_end_cam151_yolo11m_after_bundle_limited.out.log`
  - `.logs/pipe_end_cam151_yolo11m_after_bundle_limited.err.log`

Queued cam152 training:

- Dataset root: `pipe_end_detection/active_training_cam152/dataset`
- Data YAML: `pipe_end_detection/active_training_cam152/data_active.yaml`
- Run root: `pipe_end_detection/runs/pipe_end_cam152_yolo11m_after_bundle_limited/latest`
- Active output: `models/pipe_end_cam152_active/best.pt`
- Config: `imgsz=1536`, `epochs=120`, `batch=2`, `device=0`
- Initialization: `models/pipe_end_active/best.pt` after cam151 finishes, falling back to `yolo11m.pt` only if needed.
- Dataset:
  - train: 84 images, 52 positive, 32 negative, 1720 pipe-end instances
  - val: 21 images, 9 positive, 12 negative, 218 pipe-end instances
- Logs:
  - `.logs/pipe_end_cam152_yolo11m_after_bundle_limited.out.log`
  - `.logs/pipe_end_cam152_yolo11m_after_bundle_limited.err.log`

Cam152 correction after review:

- The completed `pipe_end_cam152_yolo11m_after_bundle_limited` run initialized from `models/pipe_end_active/best.pt`, so it should be treated as contaminated by cam151 for independent cam152 evaluation.
- The active file `models/pipe_end_cam152_active/best.pt` still points to that completed model until the corrected run finishes successfully.
- `apps/pipe_end_annotator/annotate_app.py` was changed so `pipe_end_cam152` no longer falls back to cam151 for prediction or first training.
- Added retraining script: `pipe_end_detection/scripts/train_cam152_yolo11m_scratch.py`.
- Corrected queue:
  - Queue PID: `.logs/pipe_end_cam152_yolo11m_scratch.queue.pid`
  - Queue log: `.logs/pipe_end_cam152_yolo11m_scratch.queue.log`
  - It waits for the active bundle segmentation training PID before using the GPU.
  - Base model: `yolo11m.pt`
  - Run root: `pipe_end_detection/runs/pipe_end_cam152_yolo11m_scratch/latest`
  - Active output after success: `models/pipe_end_cam152_active/best.pt`
  - Config: `imgsz=1536`, `epochs=120`, `batch=2`, `device=0`, heavy augmentation disabled.
  - It regenerates `pipe_end_detection/active_training_cam152/data_active.yaml` from current cam152 annotations and explicit negatives before training.
  - Logs:
    - `.logs/pipe_end_cam152_yolo11m_scratch.out.log`
    - `.logs/pipe_end_cam152_yolo11m_scratch.err.log`

### Commands for status checks

Training process check:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'tube_bundle_seg_yolo11m_sam2|pipe_end_cam152_yolo11m_scratch|detect train|segment train' } | Select-Object ProcessId,Name,CommandLine | Format-List
```

Current bundle metrics:

```powershell
Import-Csv .\tube_bundle_detection\runs\tube_bundle_yolo11m_limited_negatives\latest\results.csv | Select-Object -Last 1
```

Queued cam151 metrics after it starts:

```powershell
Import-Csv .\pipe_end_detection\runs\pipe_end_cam151_yolo11m_after_bundle_limited\latest\results.csv | Select-Object -Last 1
```

Queued cam152 metrics after it starts:

```powershell
Import-Csv .\pipe_end_detection\runs\pipe_end_cam152_yolo11m_after_bundle_limited\latest\results.csv | Select-Object -Last 1
```

GPU check:

```powershell
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

### Notebook and matching state

The notebook used to inspect YOLO detections and matching is:

- `notebooks/pipe_end_yolo_cam151_cam152.ipynb`

Known notebook/pipeline points:

- It should use the same active model paths as the annotator.
- `RUN_ID` had been set to `20260511_114027` during debugging.
- `CONF_BY_SIDE` was lowered around `0.20` to better match annotator behavior.
- The notebook should use `run_pipe_end_inference`, not raw YOLO predict, when comparing to app behavior.
- The previous problem where notebook detections looked worse than annotator detections was related to mismatched model/source/inference path assumptions.

Matching work still pending after model training:

- Improve the cam151/cam152 matching algorithm.
- Use bundle detection as a coarse reference for where the pipe package starts/ends in both cameras.
- Use pipe-end detections inside the bundle to reason about missing endpoints on either side.
- Continue validating constraints:
  - boxes should not overlap in the same place;
  - boxes should not occupy nearly the same Y height unless allowed by tolerance;
  - cam151 perspective suggests upper rows can have different apparent thickness than lower/closer rows.
- Explore gap recovery: if a gap between detections is roughly one box/tube height, rerun/search YOLO in that local region.

### Operational cautions

- Do not revert user annotations.
- The repo has many generated labels and model artifacts; avoid broad cleanup.
- Do not run multiple YOLO trainings on the GPU at the same time unless intentionally testing memory/speed.
- The app status endpoint only tracks jobs launched through the app; manual/background YOLO trainings are tracked through `.logs/*` and process checks.
- If the current bundle limited-negative run performs worse than the restored `yolo11n` model, restore/keep the previous active bundle model and consider geometric post-processing to tighten bundle boxes.

### SAM2-to-YOLO segmentation experiment

Started after the bbox bundle detector still produced poor bundle boxes. The goal is to use SAM2 as an auto-labeler for bundle masks, then train a separate YOLO segmentation model. This does not replace the bbox bundle model path.

Code added:

- `tube_bundle_detection/scripts/generate_sam2_seg_dataset.py`

SAM2 dataset generation:

- SAM model: `sam2_s.pt`
- Device for generation: `cpu`, so it can run while pipe-end training uses GPU.
- Input box labels: `tube_bundle_detection/annotation_pool/labels`
- Input images: `pipe_end_detection/annotation_pool/images`
- Output root: `tube_bundle_seg_detection/active_training_sam2_s`
- Data YAML: `tube_bundle_seg_detection/active_training_sam2_s/data_active.yaml`
- Summary: `tube_bundle_seg_detection/active_training_sam2_s/sam2_seg_dataset_summary.json`
- Visual QA overlays: `tube_bundle_seg_detection/active_training_sam2_s/overlays`
- Generated dataset:
  - train: 137 positive masks + 5 negative images
  - val: 34 positive masks + 3 negative images
  - total labels: 179
  - positive segmentation labels: 171
- Early visual check of overlays looked better than bbox-only bundle labels: masks follow the pipe package contour and the visible pipe-end side more closely.

YOLO segmentation training:

- Model: `yolo11m-seg.pt`
- Dataset: `tube_bundle_seg_detection/active_training_sam2_s/data_active.yaml`
- Run root: `tube_bundle_seg_detection/runs/tube_bundle_seg_yolo11m_sam2/latest`
- Active output path after training would have been `models/tube_bundle_seg_active/best.pt`, but this run was stopped before activation.
- Config: `imgsz=1536`, `epochs=120`, `batch=1`, `device=0`, heavy augmentation disabled.
- Queue/logs:
  - `.logs/tube_bundle_sam2_seg_dataset.out.log`
  - `.logs/tube_bundle_sam2_seg_dataset.err.log`
  - `.logs/tube_bundle_seg_yolo11m_sam2.queue.log`
  - `.logs/tube_bundle_seg_yolo11m_sam2.out.log`
  - `.logs/tube_bundle_seg_yolo11m_sam2.err.log`
- It was queued to wait for:
  - SAM2 dataset generation to finish;
  - cam151/cam152 pipe-end `yolo11m` queue to finish.
- It started successfully, but was stopped manually around epoch 53 because validation metrics degraded after an early peak and visual predictions were not usable.
- Best observed validation point was around epoch 25:
  - Mask mAP50 about `0.430`
  - Mask mAP50-95 about `0.125`
- Visual debug predictions showed:
  - `best.pt` produced many overlapping duplicate masks/boxes;
  - `last.pt` missed obvious bundles on some validation images.
- Current diagnosis: the SAM2-generated masks are too large/noisy/variable for this YOLO-seg setup. They teach an instance-seg model a huge concave scene region, not a stable clean bundle boundary. Do not just resume this exact run and expect it to recover.

Important next integration step:

- The annotator currently understands boxes, not masks.
- To use `models/tube_bundle_seg_active/best.pt` interactively, add a `tube_bundle_seg` task or a separate test script that:
  - runs YOLO-seg;
  - extracts the best mask;
  - computes a contour/bbox/ROI from the mask;
  - uses that ROI to delimit valid pipe-end detections.
- Better next experiment: generate a simplified bundle target from SAM2/box labels, such as a convex/outer hull or smoothed side boundary, instead of training on raw SAM2 masks with many notches and holes.

### Current active training after stopping bundle segmentation

- `pipe_end_cam152` corrected scratch training finished.
- Base model: `yolo11m.pt`
- It does not inherit cam151 weights.
- Run root: `pipe_end_detection/runs/pipe_end_cam152_yolo11m_scratch/latest`
- Active output: `models/pipe_end_cam152_active/best.pt`
- Logs:
  - `.logs/pipe_end_cam152_yolo11m_scratch.queue.log`
  - `.logs/pipe_end_cam152_yolo11m_scratch.out.log`
  - `.logs/pipe_end_cam152_yolo11m_scratch.err.log`
- Queue log shows start at `2026-05-18T03:01:56`.
- Best observed metrics:
  - epoch 87
  - precision about `0.945`
  - recall about `0.935`
  - mAP50 about `0.942`
  - mAP50-95 about `0.611`

### Tube bundle edge first training

- `tube_bundle_edge` was trained from the first 17 positive annotations.
- No explicit negative annotations existed for this first run.
- Training was launched through the annotator app.
- Base model: app default `yolo11n.pt`
- Run root: `tube_bundle_edge_detection/runs/tube_bundle_edge_active/latest`
- Active output: `models/tube_bundle_edge_active/best.pt`
- Best observed metrics:
  - epoch 37
  - precision about `0.683`
  - recall `1.000`
  - mAP50 about `0.913`
  - mAP50-95 about `0.603`
- Last epoch 40 had higher precision/mAP50 but lower mAP50-95:
  - precision about `0.914`
  - recall `1.000`
  - mAP50 about `0.995`
  - mAP50-95 about `0.569`
- Quick visual sanity check:
  - model generally places boxes over the pipe-end strip;
  - one tested image produced a duplicate box;
  - confidence is still low/variable on some examples.
- Treat this as a first helper model only. Add more `tube_bundle_edge` positives and explicit negatives, then retrain.

### Latest handoff - 2026-05-18

What changed most recently:

- The annotator can now resize selected boxes with visible handles on corners and sides.
  - Click a box to select it.
  - Drag a corner/side handle to make the box larger/smaller.
  - `Ctrl+S` still saves the edited label.
- The annotator was restarted on `http://127.0.0.1:8765/` with GPU training flags:
  - `.venv\Scripts\python.exe apps\pipe_end_annotator\annotate_app.py --port 8765 --no-browser --train-device 0 --train-batch 4`
- Verification done:
  - `python -m py_compile apps\pipe_end_annotator\annotate_app.py` passed.
  - The embedded JavaScript parsed successfully in Node.
  - `/api/tasks` responded from the running annotator.

What worked:

- `pipe_end_cam152` was retrained correctly from `yolo11m.pt`, not from the cam151 model.
  - Active model: `models/pipe_end_cam152_active/best.pt`
  - Best observed metrics: precision about `0.945`, recall about `0.935`, mAP50 about `0.942`, mAP50-95 about `0.611`.
- `tube_bundle_edge` is the most promising bundle-helper direction so far.
  - It detects the narrow strip where the pipe ends are visible instead of trying to box the full bundle.
  - First model used `yolo11n.pt`, `imgsz=1280`, `epochs=40`, `batch=4`, `device=0`.
  - It was trained from 17 positive examples and no explicit negatives.
  - Best observed metrics: precision about `0.683`, recall `1.000`, mAP50 about `0.913`, mAP50-95 about `0.603`.
  - Visual sanity checks were better aligned with the matching problem than the full-bundle box/segmentation experiments, but the model is still only a helper because the dataset is small.

What did not work well:

- Full `tube_bundle` one-box detection did not produce boxes precise enough for the matching algorithm.
  - The original `yolo11n` bundle run had better reported mAP50-95, about `0.637`, but the resulting boxes were still not useful enough visually.
  - The `yolo11m` run with many negatives underperformed badly.
  - The limited-negative `yolo11m` run completed 120 epochs; best mAP50-95 was about `0.467` and last mAP50-95 about `0.381`, so it is not better than the previous `yolo11n` bundle model.
- Raw SAM2 masks used as labels for YOLO segmentation did not work well.
  - The YOLO-seg model learned noisy/large scene regions, produced duplicate or missing predictions, and was stopped.
  - This should not be resumed as-is.

Current model answer:

- For the current `tube_bundle_edge` model, the active trained YOLO base is `yolo11n.pt`.
- Yes, a better model can be used. The practical next model to try is `yolo11m.pt`, but only after adding more `tube_bundle_edge` annotations and a few explicit negatives. With only 17 positives, the larger model is more likely to overfit than solve the precision issue.

Next recommended work:

- Continue annotating `tube_bundle_edge` as a vertical/narrow strip along the visible pipe-end edge, including cases where the edge runs most of the image height.
- Add explicit negative examples for `tube_bundle_edge` only when there is no visible usable pipe-end edge.
- Retrain `tube_bundle_edge` with `yolo11m.pt` after more labels are available.
- After the edge model is stable, integrate it into the cam151/cam152 matching notebook/pipeline:
  - use the edge strip to constrain valid pipe-end detections;
  - compare the ordered pipe ends inside each strip between cam151 and cam152;
  - continue the Y-height duplicate suppression and local gap recovery work for missing detections.
