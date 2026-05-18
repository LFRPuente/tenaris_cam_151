# Tube Bundle Edge Detection

This dataset is for a narrow `tube_bundle_edge` detector, not the full tube bundle.

Annotate one box per image around the visible pipe-end strip: the side/edge where the tube ends appear or should appear. If the bundle fills most of the frame, still draw only the useful edge strip, not the full bundle. If no useful pipe-end edge is visible, mark the image as a negative annotation in the annotator.

The active model path is:

```text
models/tube_bundle_edge_active/best.pt
```
