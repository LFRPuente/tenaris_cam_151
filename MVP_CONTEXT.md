# HK Sorting Table MVP Context

## 1. Purpose of This Document

This file is the working context document for the current Tenaris / HK sorting-table MVP implemented in this repository.

Its goal is to capture, in one place and with enough detail for handoff, the following:

- what the MVP is supposed to do;
- how the current backend and frontend are wired;
- which scripts, modules, notebooks, and artifact folders are part of the flow;
- what assumptions are currently encoded in the implementation;
- what camera-specific caveats exist;
- how to run and validate the MVP;
- what was done during the current build-out;
- what is still fragile, manual, or expected to change.

This document is intentionally detailed because the MVP currently sits between:

- exploratory notebook logic;
- a cleaned backend pipeline;
- local interactive tools for ROI / manual measurement / matching;
- and a client-facing viewer app.

The main risk in this project is losing context across those layers. This file exists to reduce that risk.

---

## 2. High-Level Goal of the MVP

The MVP is a client-facing viewer that shows:

- a table with sorted pipe numbering and lengths;
- the raw `cam151` image;
- the raw `cam152` image;
- the numbered tube overlays projected back onto each raw image.

The user-facing idea is:

1. A new pair of raw images arrives.
2. The backend processes both cameras with logic equivalent to the current notebooks.
3. The backend exports per-camera measurements.
4. The matcher combines both sides into a single `tube_match_latest.json`.
5. The Sorting Table MVP app reads that latest match result and renders the table plus both raw views.

The current app is not the detector itself. It is the visualization layer on top of the latest computed artifacts.

---

## 3. Current Repository Components That Matter for the MVP

### 3.1 Client-facing MVP viewer

- [run_sorting_table_mvp.py](/c:/Users/luis_/Desktop/tenaris_cam_152/run_sorting_table_mvp.py)
- [run_sorting_table_mvp_bg.py](/c:/Users/luis_/Desktop/tenaris_cam_152/run_sorting_table_mvp_bg.py)
- [sorting_table_mvp_proc.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/sorting_table_mvp_proc.py)
- [sorting_table_mvp.html](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/sorting_table_mvp.html)

### 3.2 Clean backend pipeline extracted from notebooks

- [process_tube_pair.py](/c:/Users/luis_/Desktop/tenaris_cam_152/process_tube_pair.py)
- [src/tenaris_tube_pipeline/__init__.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/__init__.py)
- [src/tenaris_tube_pipeline/config.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/config.py)
- [src/tenaris_tube_pipeline/camera.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/camera.py)
- [src/tenaris_tube_pipeline/pair.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/pair.py)
- [src/tenaris_tube_pipeline/notebook_style_detection.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/notebook_style_detection.py)

### 3.3 Existing ROI / matching / measurement ecosystem

- [run_manual_tube_measure.py](/c:/Users/luis_/Desktop/tenaris_cam_152/run_manual_tube_measure.py)
- [run_manual_tube_measure_bg.py](/c:/Users/luis_/Desktop/tenaris_cam_152/run_manual_tube_measure_bg.py)
- [run_tube_matcher.py](/c:/Users/luis_/Desktop/tenaris_cam_152/run_tube_matcher.py)
- [run_tube_matcher_bg.py](/c:/Users/luis_/Desktop/tenaris_cam_152/run_tube_matcher_bg.py)
- [src/cam151_ref_detection/manual_tube_measure_proc.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/manual_tube_measure_proc.py)
- [src/cam151_ref_detection/tube_matcher_proc.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/tube_matcher_proc.py)
- [src/cam151_ref_detection/homography_preview.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/homography_preview.py)
- [src/cam151_ref_detection/roi_store.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/roi_store.py)
- [src/cam151_ref_detection/warp_roi_picker_proc.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/warp_roi_picker_proc.py)
- [src/cam151_ref_detection/ref_line_picker_proc.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/ref_line_picker_proc.py)
- [src/cam151_ref_detection/scale_line_picker_proc.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/scale_line_picker_proc.py)

### 3.4 Source notebooks

- [tube_detection_step_by_step_cam151.ipynb](/c:/Users/luis_/Desktop/tenaris_cam_152/notebooks/tube_detection_step_by_step_cam151.ipynb)
- [tube_detection_step_by_step_cam152.ipynb](/c:/Users/luis_/Desktop/tenaris_cam_152/notebooks/tube_detection_step_by_step_cam152.ipynb)

---

## 4. Core MVP Data Flow

The current MVP flow is:

1. Raw images are available, typically:
   - `test_images/cam_151_202604022.jpeg`
   - `test_images/cam_152_202604022.jpeg`
2. Manual ROI TOMLs are available for each image.
3. The clean backend pipeline processes each camera independently using notebook-style logic.
4. Each camera export is written to `artifacts/tube_matcher_inputs/`.
5. The pair matcher writes:
   - a historical JSON;
   - a historical XLSX;
   - `tube_match_latest.json`;
   - `tube_match_latest.xlsx`.
6. The Sorting Table MVP server reads the latest match result and resolves:
   - the correct per-camera datasets;
   - the raw image paths;
   - the manual ROI paths;
   - the homography needed to project warp-space measurements back to raw coordinates.
7. The app generates an inline HTML page with:
   - the table rows;
   - two image states;
   - per-camera marker coordinates;
   - front-end interaction logic.
8. The browser renders:
   - the table;
   - the raw image assets;
   - the SVG overlays.

Important consequence:

- The app is always a consumer of the latest backend artifacts.
- It is not performing tube detection in the browser.

---

## 5. Clean Backend Pipeline: Why It Exists

Originally, the actual detection logic lived mainly in notebooks.

That is useful for exploration, but risky for production because notebooks:

- keep implicit state;
- allow out-of-order cell execution;
- are harder to diff and maintain;
- are not a good runtime integration layer for a client-facing app.

The package [src/tenaris_tube_pipeline](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline) was created as a clean extraction layer so that:

- future processing can run from Python modules;
- notebooks can remain the experimentation surface;
- the app can depend on stable exported artifacts;
- changes in notebooks can later be ported intentionally into the pipeline.

The working design principle is:

- notebooks are the exploration source;
- the clean pipeline is the backend execution source;
- the MVP app reads backend outputs.

---

## 6. Clean Backend Pipeline Structure

### 6.1 Entry point

[process_tube_pair.py](/c:/Users/luis_/Desktop/tenaris_cam_152/process_tube_pair.py) is the CLI entry point.

It:

- accepts both image paths;
- optionally accepts explicit ROI paths;
- optionally accepts output locations;
- builds per-camera configs;
- calls `process_tube_pair(...)`;
- prints tube counts and output paths.

CLI arguments:

- `--cam151-image`
- `--cam152-image`
- `--cam151-roi`
- `--cam152-roi`
- `--artifact-root`
- `--matcher-input-dir`
- `--match-output-dir`
- `--output-stem`

### 6.2 Config layer

[src/tenaris_tube_pipeline/config.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/config.py) defines:

- `CameraPipelineConfig`
- `PipelineOutputConfig`

It also provides defaults for:

- backend artifact root;
- matcher input directory;
- final match output directory;
- default ROI resolution from `manual_rois/<stem>_rois.toml`.

### 6.3 Per-camera processing

[src/tenaris_tube_pipeline/camera.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/camera.py) does the camera-side work.

Responsibilities:

- load ROI TOML;
- run notebook-style detection via `detect_tubes_like_notebook(...)`;
- build tube measurements in the export format expected by the matcher;
- export those measurements through existing matcher tooling;
- capture metadata that links the export back to the backend pipeline.

### 6.4 Pair processing

[src/tenaris_tube_pipeline/pair.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/pair.py) orchestrates both cameras.

Responsibilities:

- process cam151;
- process cam152;
- load both exported datasets;
- build the merged match payload;
- write historical and latest match artifacts.

### 6.5 Notebook-style detection

[src/tenaris_tube_pipeline/notebook_style_detection.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/notebook_style_detection.py) is the crucial extraction layer.

It ports the practical notebook logic, including:

- homography creation;
- ROI warping;
- tube stack range estimation;
- dark-run profiling;
- seam / periodic peak detection;
- periodic gap filling;
- smoothed x-position detection;
- notebook-style reference line generation from `mark_02` / `mark_03`.

This module intentionally reuses low-level helpers from the older codebase, especially from `tube_detection_preview.py`, because the priority was parity with notebook behavior, not a total rewrite.

---

## 7. Why `mark_02` / `mark_03` Matter

One important discovery during implementation:

- the clean backend needed to follow the notebook reference-line behavior exactly;
- the most reliable parity came from using the notebook-style line defined by `mark_02` and `mark_03`;
- this reference line is encoded in `notebook_style_detection.py`.

This matters because distance calculations and the projected measurement positions depend on the chosen reference line.

If this line changes in future notebook work, the clean backend probably also needs to change.

---

## 8. Match Artifacts the MVP Consumes

The app primarily relies on:

- `artifacts/tube_matching/tube_match_latest.json`
- `artifacts/tube_matcher_inputs/cam151_tube_measurements_latest.json`
- `artifacts/tube_matcher_inputs/cam152_tube_measurements_latest.json`

For the current validated pair, the historical source in the match payload is:

- `cam151_tube_measurements_20260427_184007.json`
- `cam152_tube_measurements_20260427_184007.json`

The current `tube_match_latest.json` summary indicates:

- `matched = 45`
- `left_only = 0`
- `right_only = 0`

This is the current canonical pair used during MVP setup.

---

## 9. Camera-Specific Caveats

### 9.1 cam151

The working image is:

- `test_images/cam_151_202604022.jpeg`

The working ROI is:

- `manual_rois/cam_151_202604022_rois.toml`

### 9.2 cam152

The working image is:

- `test_images/cam_152_202604022.jpeg`

The important caveat is the ROI source.

During validation, the local repo ROI:

- `manual_rois/cam_152_202604022_rois.toml`

was not sufficient for the notebook-style pipeline because the working configuration needed the reference points used in the notebook flow.

The working ROI path used during validation was:

- `C:\Users\luis_\Downloads\cam_152_202604022.jpeg_rois_export (1).toml`

This is a critical operational caveat for the MVP:

- the current validated cam152 flow depends on a ROI TOML outside the repo;
- if that file is moved, deleted, or replaced, the current validated behavior may break;
- this should eventually be normalized into a repo-owned ROI artifact if possible.

### 9.3 Mirror behavior for cam152

Another important caveat is mirror behavior.

The current logic is aligned with the notebook/backend rule:

- `cam_152.jpeg` base variant is already mirrored on disk;
- `cam_152_*` variants need additional backend mirror handling.

This logic is encoded in:

- [sorting_table_mvp_proc.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/sorting_table_mvp_proc.py)
- [tube_detection_preview.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/tube_detection_preview.py)

The MVP viewer had to be aligned to this same rule so that projected markers land in the same place as the backend/notebook logic.

---

## 10. Sorting Table MVP Server Architecture

The client-facing MVP server lives in:

- [sorting_table_mvp_proc.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/sorting_table_mvp_proc.py)

### 10.1 What the server does

The server:

- resolves the latest match result;
- resolves per-camera measurement datasets;
- resolves raw image files;
- resolves ROI TOMLs;
- reconstructs the warp-to-raw inverse mapping;
- projects measurement points back to raw coordinates;
- builds the HTML state;
- serves one HTML page and two raw-image assets.

### 10.2 Important behavior

On every request to `/` or `/index.html`, the server rebuilds a fresh runtime snapshot.

That means:

- it does not keep one static startup snapshot forever;
- if `tube_match_latest.json` changes, a page refresh will pick it up.

This is very useful for MVP iteration because the frontend view tracks the latest backend result without a full app rebuild.

### 10.3 What the server serves

The server serves:

- the HTML UI at `/`;
- raw cam151 asset at `/asset/cam151-raw...`;
- raw cam152 asset at `/asset/cam152-raw...`.

### 10.4 Current live URL tracking

The background launcher writes the active URL to:

- [artifacts/sorting_table_mvp/live_url_current.txt](/c:/Users/luis_/Desktop/tenaris_cam_152/artifacts/sorting_table_mvp/live_url_current.txt)

At the time this document was written, that file contained:

- `http://127.0.0.1:58597/`

---

## 11. How Marker Projection Works in the MVP

The app does not use arbitrary hand-placed points.

It:

1. Reads the per-tube measurement export.
2. Pulls the warp-space tube coordinate:
   - usually `x_start_raw_warp` or equivalent;
   - plus `y_center_warp`.
3. Rebuilds the homography from the raw image plus ROI data.
4. Builds the final transform used for the warp.
5. Inverts that transform.
6. Projects the warp-space point back to raw-image coordinates.
7. Applies cam152 mirror handling when needed.

This is why the overlay is tied to the backend geometry, not to a frontend-only approximation.

---

## 12. Frontend UI Behavior of the MVP

The UI template lives in:

- [sorting_table_mvp.html](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/sorting_table_mvp.html)

### 12.1 Main layout

The page includes:

- title and summary cards;
- a left-side table;
- two raw-image viewers;
- overlay markers rendered with SVG.

### 12.2 Table behavior

The table shows:

- `Pipe #`
- `Pipe Length (In)`
- `Pipe Length (Ft)`

Click behavior:

- clicking a row selects the tube;
- the selected tube is highlighted in the table;
- the selected tube is highlighted in both image viewers;
- clicking a table row also zooms / centers both viewers on that same tube.

### 12.3 Viewer behavior

Each viewer supports:

- `+` zoom;
- `-` zoom;
- `100%` reset;
- wheel zoom;
- middle-button drag pan;
- initial auto-focus to the marker zone on first render.

### 12.4 Selection behavior

Current selection behavior includes:

- selected tube bubble turns black;
- selection is synchronized across table + both images;
- when selected from the table, both views focus on that tube.

### 12.5 Current label direction rules

Current per-camera rule:

- `cam151`: number bubble is placed to the right of the detected point;
- `cam152`: number bubble is placed to the left of the detected point.

This was chosen for readability in the current camera geometry.

### 12.6 Current bubble styling rule

Even though adaptive color logic exists in the backend, the current frontend presentation was intentionally normalized so that:

- all number bubbles use one fixed visual color;
- the selected bubble turns black.

This was done because variable bubble colors made users think the colors represented different semantic categories.

---

## 13. Important MVP UI Decisions Made During Iteration

The current UI behavior is the result of several iterations.

The following decisions were made deliberately:

### 13.1 Numbers should not imply different categories

Result:

- all bubbles are now one fixed color;
- only the selected bubble changes color.

### 13.2 Number placement should prioritize human readability over geometric purity

Result:

- the point itself remains the backend-projected truth;
- the bubble can be shifted horizontally for legibility.

### 13.3 Row click should become a navigation shortcut

Result:

- clicking a table row also zooms and centers both viewers.

### 13.4 The app should open near the interesting content

Result:

- initial viewer scroll is focused on the bounding region containing markers.

---

## 14. Current Known Working Commands

### 14.1 Process the validated current pair

This is the validated command that was used to drive the current pair while explicitly using the working cam152 ROI from `Downloads`:

```powershell
.\.venv\Scripts\python.exe process_tube_pair.py `
  --cam151-image test_images\cam_151_202604022.jpeg `
  --cam152-image test_images\cam_152_202604022.jpeg `
  --cam152-roi "C:\Users\luis_\Downloads\cam_152_202604022.jpeg_rois_export (1).toml" `
  --output-stem backend_current_pair
```

### 14.2 Validation command with isolated artifact output

This was used earlier to validate parity without overwriting the default latest files:

```powershell
.\.venv\Scripts\python.exe process_tube_pair.py `
  --cam151-image test_images\cam_151_202604022.jpeg `
  --cam152-image test_images\cam_152_202604022.jpeg `
  --cam152-roi "C:\Users\luis_\Downloads\cam_152_202604022.jpeg_rois_export (1).toml" `
  --artifact-root artifacts\backend_tube_pipeline_validation `
  --matcher-input-dir artifacts\backend_tube_pipeline_validation\tube_matcher_inputs `
  --match-output-dir artifacts\backend_tube_pipeline_validation\tube_matching `
  --output-stem backend_validation_current_pair
```

### 14.3 Launch the client-facing MVP app

Foreground:

```powershell
.\.venv\Scripts\python.exe run_sorting_table_mvp.py --port 58597 --no-browser
```

Background:

```powershell
.\.venv\Scripts\python.exe run_sorting_table_mvp_bg.py --port 58597 --url-file artifacts/sorting_table_mvp/live_url_current.txt
```

---

## 15. Current Live MVP App State

At the time of writing this context file:

- the active background app URL file points to `http://127.0.0.1:58597/`;
- the latest match JSON still reports `45` matched tubes;
- the app includes the latest interaction improvements made during the MVP iteration:
  - per-camera bubble direction;
  - single bubble color;
  - black selected bubble;
  - row-click focus/zoom;
  - initial auto-focus to the marker area.

---

## 16. Validation Summary That Was Achieved

The clean backend pipeline was validated against the current pair.

Validated outcome:

- cam151 tube count: `45`
- cam152 tube count: `45`
- matched: `45`
- left_only: `0`
- right_only: `0`

Additional validation that was done earlier:

- projected positions matched the latest exports in practical terms;
- per-camera exported positions were consistent with the latest matcher inputs;
- distance differences versus current exports were effectively floating-point noise.

This means the clean backend pipeline is currently good enough to feed the MVP app for the validated pair.

---

## 17. Relationship Between the MVP App and the Older Tools

The MVP viewer does not replace the older tools.

Current tool roles are:

- ROI / reference pickers:
  - define and update manual ROI, reference-line, and scale inputs from the notebook/web picker flow;
- manual measure app:
  - inspect or adjust measurement-oriented logic / references;
- matcher app:
  - inspect pairing / matching outputs;
- clean backend pipeline:
  - process the pair from raw images to latest match artifact;
- sorting table MVP:
  - client-facing visualization layer.

These tools are related, but not interchangeable.

---

## 18. What Still Depends on Manual Knowledge

The MVP is working, but still depends on a few human-known facts:

### 18.1 Working cam152 ROI path

The validated cam152 ROI is currently external to the repo.

### 18.2 Notebook parity is still a deliberate maintenance task

If the notebooks change meaningfully, those changes are not automatically reflected in the clean backend package.

That means:

- notebook changes must still be ported into `src/tenaris_tube_pipeline`;
- validation should be re-run after those ports.

### 18.3 The MVP app assumes latest artifacts are trustworthy

The app does not independently verify whether `tube_match_latest.json` is stale, wrong, or mismatched to a different image pair.

It assumes the backend artifacts it consumes are the correct current truth.

---

## 19. Recommended Future Direction

### 19.1 Move the working cam152 ROI into a stable repo-owned location

This is one of the highest-value cleanup items.

### 19.2 Make notebooks call the clean backend package where possible

That would help flip the source of truth from “copy notebook logic into Python” to “notebook uses the Python pipeline.”

### 19.3 Add parity tests or validation scripts

Especially for:

- tube count;
- match count;
- sample coordinate parity;
- distance parity.

### 19.4 Consider formalizing a notebook-to-backend synchronization workflow

A future skill or internal workflow could help enforce:

- identify changed notebook cells;
- port logic to `src/tenaris_tube_pipeline`;
- run parity checks;
- update this context file when behavior changes.

### 19.5 Normalize the UI rules if client expectations harden

The current overlay UX was tuned iteratively for readability.

If the MVP graduates into a more permanent product, it may make sense to freeze:

- final bubble size;
- final bubble placement rules;
- final selection behavior;
- final zoom/focus rules;
- exact visual language.

---

## 20. Summary of the Current MVP in One Paragraph

The current HK Sorting Table MVP is a local client-facing web app that reads the latest tube-matching artifacts, displays a pipe-length table plus both raw camera images, projects backend-computed tube numbering onto those raw views, and supports synchronized selection, zoom, and focus across the table and both images. The backend feeding it is a clean Python pipeline extracted from the current notebooks, validated on the `cam_151_202604022.jpeg` / `cam_152_202604022.jpeg` pair with `45` matched tubes, with the main remaining operational caveat being the currently external working ROI file for `cam152`.

---

## 21. Files Most Important to Read First

If someone needs to re-enter this project quickly, the best reading order is:

1. [MVP_CONTEXT.md](/c:/Users/luis_/Desktop/tenaris_cam_152/MVP_CONTEXT.md)
2. [process_tube_pair.py](/c:/Users/luis_/Desktop/tenaris_cam_152/process_tube_pair.py)
3. [src/tenaris_tube_pipeline/notebook_style_detection.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/notebook_style_detection.py)
4. [src/tenaris_tube_pipeline/camera.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/camera.py)
5. [src/tenaris_tube_pipeline/pair.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/tenaris_tube_pipeline/pair.py)
6. [src/cam151_ref_detection/sorting_table_mvp_proc.py](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/sorting_table_mvp_proc.py)
7. [src/cam151_ref_detection/sorting_table_mvp.html](/c:/Users/luis_/Desktop/tenaris_cam_152/src/cam151_ref_detection/sorting_table_mvp.html)
8. [notebooks/tube_detection_step_by_step_cam151.ipynb](/c:/Users/luis_/Desktop/tenaris_cam_152/notebooks/tube_detection_step_by_step_cam151.ipynb)
9. [notebooks/tube_detection_step_by_step_cam152.ipynb](/c:/Users/luis_/Desktop/tenaris_cam_152/notebooks/tube_detection_step_by_step_cam152.ipynb)

---

## 22. Maintenance Note

Whenever any of the following changes, this document should be updated:

- image pair used as the validated reference;
- ROI source paths;
- notebook reference-line logic;
- backend mirror behavior;
- match artifact schema;
- Sorting Table MVP interaction model;
- launcher commands;
- live operational caveats.

---

## 23. Capture History Extension (2026-04-27)

The Sorting Table MVP now also supports a camera-driven pair workflow on top of the existing latest-artifact viewer model.

Current behavior:

- the main MVP view includes a `Descargar y procesar` action;
- that action downloads a fresh pair from `cam151` and `cam152`;
- the pair is processed automatically with the clean backend pipeline;
- the downloaded raw images and the full processing outputs are stored inside a dedicated per-run folder;
- the app also exposes a separate `/history` view where runs can be filtered by capture date and reopened later.

Current per-run storage convention:

- `artifacts/capture_history/<run_id>/raw/`
- `artifacts/capture_history/<run_id>/processing/backend_tube_pipeline/`
- `artifacts/capture_history/<run_id>/processing/tube_matcher_inputs/`
- `artifacts/capture_history/<run_id>/processing/tube_matching/`
- `artifacts/capture_history/<run_id>/manifest.json`

Operational rule:

- opening a historical run only changes what the MVP viewer renders;
- it does not globally retarget every tool to that historical run;
- the global `latest` artifacts still represent the most recent successfully downloaded and processed run.

Current ROI rule for downloaded pairs:

- downloaded pairs currently reuse fixed ROI/reference/scale inputs instead of per-run ROI authoring;
- `cam151` uses `manual_rois/_frozen_defaults/cam_151_202604022_rois.toml`;
- `cam152` uses `manual_rois/_frozen_defaults/cam_152_202604022_rois.toml`.

This is an intentional temporary rule. ROI / reference / scale editing for newly downloaded pairs remains a future extension.

---

## 24. Pipe-End YOLO Status (2026-05-10)

YOLO `pipe_end` work now exists in this repository, but it is not yet the default detector for the MVP.

Current rule:

- production MVP processing uses the classical notebook-style detector by default;
- YOLO can be enabled only as an experiment with `PIPE_END_YOLO_ENABLED=1`;
- the immediate integration path is notebook-first validation, not direct production replacement.

Relevant files:

- `src/pipe_end_yolo/inference.py`
- `apps/pipe_end_annotator/annotate_app.py`
- `pipe_end_detection/capture_pipe_end_dataset.py`
- `pipe_end_detection/scripts/prepare_yolo_dataset.py`
- `pipe_end_detection/scripts/train_yolo.py`
- `pipe_end_detection/scripts/predict_unlabeled.py`
- `YOLO_MVP_INTEGRATION_PLAN.md`
- `PIPE_END_YOLO_TRAINING_PLAN.md`
- `GITHUB_HANDOFF.md`

Operational meaning:

- downloaded MVP runs should remain stable with the classical detector unless YOLO is explicitly enabled;
- YOLO should first be run inside the cam151/cam152 notebooks on the same homography warp images used by the existing detector;
- the notebook comparison must check tube count, duplicate detections, missed detections, ordering, and cam151/cam152 match consistency;
- only after that validation should YOLO be allowed to drive `x_start_list` for MVP processing.

Large data policy:

- raw captures, annotation images, generated datasets, training runs, and `.pt` model weights are intentionally ignored by normal Git;
- if all data must move through GitHub, use Git LFS or release artifacts;
- labels and small metadata can be versioned normally when useful.

