# Tube YOLO Annotation

Proyecto limpio para anotar y entrenar detecciones `pipe_end` y `tube_bundle`.

## Anotacion

Imagenes:

```text
annotation_pool/images/cam151
annotation_pool/images/cam152
```

Labels YOLO `pipe_end`:

```text
annotation_pool/labels/cam151
annotation_pool/labels/cam152
```

Clase unica `pipe_end`:

```text
0 pipe_end
```

Clase unica `tube_bundle`:

```text
0 tube_bundle
```

Clase unica `tube_bundle_edge`:

```text
0 tube_bundle_edge
```

Los labels `tube_bundle` se guardan separados de `pipe_end`:

```text
tube_bundle_detection/annotation_pool/labels/cam151
tube_bundle_detection/annotation_pool/labels/cam152
```

Los labels `tube_bundle_edge` tambien se guardan separados:

```text
tube_bundle_edge_detection/annotation_pool/labels/cam151
tube_bundle_edge_detection/annotation_pool/labels/cam152
```

## App local de anotacion

```powershell
python apps\pipe_end_annotator\annotate_app.py --port 8765 --no-browser
```

Uso:

- selector `Task`:
  - `pipe_end | cam151 model`: labels existentes de `pipe_end` para cam151 y entrena `models/pipe_end_active/best.pt`.
  - `pipe_end | cam152 model`: solo muestra cam152 y entrena `models/pipe_end_cam152_active/best.pt`.
  - `tube_bundle | one box`: un solo box del paquete de tubos por imagen y entrena `models/tube_bundle_active/best.pt`.
  - `tube_bundle_edge | pipe-end strip`: una caja sobre la franja donde estan las puntas de los tubos; entrena `models/tube_bundle_edge_active/best.pt`.
- arrastra con el mouse para dibujar una caja
- click sobre una caja para seleccionarla
- arrastra los vertices/lados de la caja seleccionada para hacer resize fino
- `Delete` borra la caja seleccionada
- `Ctrl+S` guarda el label YOLO
- flechas izquierda/derecha cambian de imagen
- filtro lateral permite ver `cam151`, `cam152`, pendientes o anotadas
- `Bad warp` marca una imagen para excluirla del dataset
- `Negative image` marca explicitamente una imagen `pipe_end` de cam151/cam152 o `tube_bundle_edge` como negativa; se entrena con label vacio y no se confunde con imagenes pendientes de anotar
- `Run model on this image` corre el modelo activo solo en la imagen actual y carga las cajas para corregirlas
- `Load AI boxes` carga predicciones YOLO para corregirlas manualmente

## Preparar dataset YOLO

```powershell
python pipe_end_detection\scripts\prepare_yolo_dataset.py `
  --images-root pipe_end_detection\annotation_pool\images `
  --labels-root pipe_end_detection\annotation_pool\labels `
  --output-root pipe_end_detection\dataset `
  --allow-missing-labels
```

Las imagenes marcadas como `Bad warp` se excluyen automaticamente.

## Active learning

1. Anota manualmente al menos `15-25` imagenes buenas.
2. Entrena primer modelo desde el boton `Train AI`.
3. Genera predicciones con `Generate predictions`.
4. En la app, abre una imagen con badge `AI`, usa `Load AI boxes`, corrige las cajas y guarda.

## Entrenar con Ultralytics

```powershell
yolo detect train data=pipe_end_detection\data.yaml model=yolo11n.pt imgsz=1280 epochs=100 batch=8
```
