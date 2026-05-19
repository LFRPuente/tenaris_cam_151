# Tenaris Camera MVP

Local MVP for downloading camera pairs, processing tube measurements, matching both camera views, and visualizing the sorting table.

## Main Runtime

- `run_sorting_table_mvp.py`: foreground launcher for the MVP.
- `run_sorting_table_mvp_bg.py`: background launcher for the MVP.
- `process_tube_pair.py`: CLI entrypoint for processing one raw cam151/cam152 pair.
- `config.example.json`: safe camera config template; copy to `config.json` locally and fill credentials/IPs.
- `src/tenaris_tube_pipeline/`: clean backend processing pipeline extracted from the notebooks.
- `src/cam151_ref_detection/sorting_table_mvp_proc.py`: MVP server, history, capture endpoint, image/table state.
- `src/cam151_ref_detection/capture_history.py`: camera download, capture history, and capture processing orchestration.

## Calibration And Internal Tools

- `notebooks/tube_detection_step_by_step_cam151.ipynb`
- `notebooks/tube_detection_step_by_step_cam152.ipynb`
- `src/cam151_ref_detection/warp_roi_picker_proc.py`
- `src/cam151_ref_detection/scale_line_picker_proc.py`
- `src/cam151_ref_detection/ref_line_picker_proc.py`
- `src/cam151_ref_detection/homography_preview.py`

## Pipe-End YOLO Work

- `pipe_end_detection/capture_pipe_end_dataset.py`: captures clean wide images for YOLO annotation.
- `apps/pipe_end_annotator/annotate_app.py`: local browser app for drawing/correcting YOLO boxes.
- `pipe_end_detection/scripts/prepare_yolo_dataset.py`: prepares YOLO dataset splits.
- `pipe_end_detection/scripts/train_yolo.py`: trains the YOLO model.
- `pipe_end_detection/scripts/predict_unlabeled.py`: generates AI boxes for review.
- `pipe_end_detection/ANNOTATION_GUIDE.md`: labeling rules.
- `src/pipe_end_yolo/inference.py`: experimental runtime YOLO inference hook for the processing pipeline.

Current runtime default:

- The backend enables YOLO `pipe_end` by default unless `PIPE_END_YOLO_ENABLED=0` is set.
- The classical/notebook-style detector remains available as a fallback.
- Runtime YOLO post-processing includes overlap/duplicate suppression, gap recovery, SAM-boundary edge recovery, large-box Sobel-Y splitting, lateral outlier confidence filtering, and Sobel-X edge refinement.
- SAM is currently used as a prompted boundary segmenter with base `sam2.1_s.pt`; it is not fine-tuned.
- The matcher can use SAM-normalized bundle height when both camera datasets export SAM Y-bounds.

Runtime controls:

- `PIPE_END_YOLO_MODEL`: optional explicit model path.
- `PIPE_END_YOLO_DEVICE`: optional Ultralytics device, for example `0` or `cpu`.
- `PIPE_END_YOLO_CONF`: optional global confidence threshold.
- `PIPE_END_YOLO_CONF_CAM151`: optional cam151-specific confidence threshold.
- `PIPE_END_YOLO_CONF_CAM152`: optional cam152-specific confidence threshold.
- `PIPE_END_YOLO_IOU`: optional YOLO IoU threshold.
- `PIPE_END_YOLO_VERTICAL_DUPLICATE_Y_OVERLAP`: suppresses boxes that occupy the same vertical band.
- `PIPE_END_YOLO_GAP_RECOVERY_ENABLED`: enables/disables the second-pass YOLO search in vertical gaps.
- `PIPE_END_YOLO_GAP_RECOVERY_CONF`: confidence threshold for the gap-recovery pass.
- `PIPE_END_YOLO_EDGE_GAP_RECOVERY_ENABLED`: enables/disables recovery near SAM top/bottom bounds.
- `PIPE_END_YOLO_LARGE_BOX_SPLIT_ENABLED`: enables/disables Sobel-Y splitting for locally oversized boxes.
- `PIPE_END_YOLO_FAR_X_CONF_FILTER_ENABLED`: requires higher confidence for lateral X outliers.
- `PIPE_END_YOLO_ENABLED=0`: disables YOLO and falls back to the classical detector.

Large local data is intentionally ignored:

- `artifacts/`
- `pipe_end_detection/captures/`
- `pipe_end_detection/annotation_pool/images/`
- `pipe_end_detection/dataset/`
- `pipe_end_detection/runs/`
- `models/`

Important docs:

- `MVP_CONTEXT.md`: detailed context for the MVP and processing pipeline.
- `YOLO_MVP_INTEGRATION_PLAN.md`: how YOLO should be validated and integrated into the MVP.
- `PIPE_END_YOLO_TRAINING_PLAN.md`: how to keep annotating and training the `pipe_end` model.
- `GITHUB_HANDOFF.md`: how to move this repo to another computer/GPU.

## Next Development Rule

When changing pipe-end logic, validate both cameras on representative captures and inspect the match strategy. If SAM bounds are available for both camera exports, the matcher should report `sam_bundle_normalized_vertical_slots`; otherwise it falls back to local vertical-slot matching.
