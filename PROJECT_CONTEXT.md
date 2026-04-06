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
