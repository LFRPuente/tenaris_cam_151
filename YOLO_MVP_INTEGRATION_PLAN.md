# YOLO + MVP Integration Plan

## Goal

Use YOLO `pipe_end` detections as a production input to the Sorting Table MVP, while keeping the current notebook-style tube pipeline available as the baseline/fallback.

## Current State

- The MVP downloads camera pairs, processes both cameras, writes capture history, and renders the latest processed run.
- The MVP production default is still the classical/notebook-style detector.
- The YOLO `pipe_end` runtime hook exists in `src/pipe_end_yolo/inference.py`, but it is disabled by default.
- Enable YOLO only for experiments with `PIPE_END_YOLO_ENABLED=1`.
- The classical detector currently provides homography, scale, reference geometry, and tube starts.
- Runtime inference lives in `src/pipe_end_yolo/inference.py`.
- The YOLO annotator, training, prediction scripts, and dataset preparation scripts have been migrated as code into this repo.
- Large datasets, generated runs, and model weights remain local/ignored unless Git LFS or external dataset storage is configured.

## YOLO Code Now In This Repo

The YOLO annotation and training code has been migrated into this repo.

Current structure:

```text
src/pipe_end_yolo/
  __init__.py
  inference.py

apps/pipe_end_annotator/
  annotate_app.py

pipe_end_detection/
  capture_pipe_end_dataset.py
  scripts/prepare_yolo_dataset.py
  scripts/train_yolo.py
  scripts/predict_unlabeled.py
  scripts/rewarp_pool.py
  ANNOTATION_GUIDE.md
  classes.txt
  data.yaml
```

Large files remain local or use Git LFS:

- `pipe_end_detection/captures/`
- `pipe_end_detection/annotation_pool/images/`
- `dataset*/`
- `runs/`
- `active_training/`
- `models/*.pt`

Labels and small metadata should be versioned:

- `annotation_pool/labels/**/*.txt`
- `image_status.json`
- capture manifests if they are small enough and do not contain secrets.

## Runtime Flow

1. User clicks `Download and Process` in the MVP.
2. `capture_history.capture_and_process_pair()` downloads cam151 and cam152.
3. The existing pipeline processes the pair and writes tube matching artifacts.
4. In experimental mode, a YOLO step runs on the same homography warp used by the pipeline.
5. YOLO runs inference on each warped image.
6. Predictions are written into the run folder:

```text
artifacts/capture_history/<run_id>/pipe_end_yolo/
  cam151_pipe_end_predictions.json
  cam152_pipe_end_predictions.json
  cam151_pipe_end_overlay.jpg
  cam152_pipe_end_overlay.jpg
```

7. The run manifest stores paths and summary:

```json
{
  "pipe_end_yolo": {
    "enabled": true,
    "model_path": "models/pipe_end_active/best.pt",
    "cam151": {
      "prediction_count": 45,
      "predictions_path": "...",
      "overlay_path": "..."
    },
    "cam152": {
      "prediction_count": 45,
      "predictions_path": "...",
      "overlay_path": "..."
    }
  }
}
```

## MVP UI Changes

- Add a toggle: `Tube Overlay` / `YOLO Pipe-End Overlay`.
- Show YOLO detection count per camera in the summary cards.
- On each image, render YOLO boxes or center-line markers on top of the current raw/warped view.
- If YOLO is missing or fails, the MVP should still load the classical processing result.

## Matching Strategy

Phase 1:

- YOLO is evaluated in notebooks and diagnostic scripts first.
- The MVP table continues to use the classical matched tube output.

Phase 2:

- Convert YOLO `pipe_end` boxes to per-tube start positions in warp coordinates.
- Compare YOLO count and classical count.
- Save diagnostics:
  - missing YOLO detections;
  - duplicate YOLO detections;
  - tube index alignment errors.

Phase 3:

- Use YOLO pipe-end positions as the primary source for tube start detection.
- Keep classical detection as fallback if YOLO confidence/count fails validation.

## Validation Rules Before YOLO Drives The Table

- Per camera count must be within expected range.
- Boxes must not heavily overlap.
- Vertical order must be monotonic after sorting.
- Box centers must stay inside the tube stack ROI.
- Median pitch must be plausible for that camera/current ROI.
- cam151/cam152 matched count must be consistent.

## Immediate Next Steps

1. Add notebook cells to run YOLO on the current warp and compare against classical tube starts.
2. Keep improving `src/pipe_end_yolo/inference.py`:

```python
run_pipe_end_inference(image_path, model_path, output_dir) -> PipeEndInferenceResult
```

3. Continue active-learning annotation and training using `PIPE_END_YOLO_TRAINING_PLAN.md`.
4. Add a model registry/version field so the MVP can show which model generated each run.
5. Add direct MVP overlay rendering for YOLO boxes, separate from the current matched tube markers.
6. Keep model files out of normal Git until Git LFS or release storage is configured.
