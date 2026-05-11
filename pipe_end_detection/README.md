# Pipe End Detection

Este paquete deja preparado el dataset para un modelo de deteccion de objetos tipo YOLOv11 con una sola clase:

- `pipe_end`

El objetivo es detectar el extremo visible de cada tubo gris de inventario mediante bounding boxes ajustadas.

## Estructura

- `ANNOTATION_GUIDE.md`
  Guia de anotacion operativa.
- `classes.txt`
  Lista de clases.
- `data.yaml`
  Configuracion YOLOv11.
- `capture_pipe_end_dataset.py`
  Descarga pares de camaras con `config.json`, genera imagenes homografiadas y las coloca en el pool de anotacion.
- `prepare_yolo_dataset.py`
  Script para consolidar imagenes y labels en formato YOLO.
- `captures/`
  Corridas descargadas para dataset: raw, warped, overlays y manifest.
- `annotation_pool/`
  Imagenes warp listas para anotar y labels YOLO vacios.
- `dataset/`
  Salida lista para entrenamiento.

## Captura para anotacion

El flujo para YOLO debe anotar las imagenes ya homografiadas, no las raw. Para capturar un par:

```powershell
python pipe_end_detection\capture_pipe_end_dataset.py
```

Salida por corrida:

```text
pipe_end_detection/
  captures/
    runs/
      <run_id>/
        raw/
        warped/
        overlays/
        manifest.json
  annotation_pool/
    images/
      cam151/
      cam152/
    labels/
      cam151/
      cam152/
```

Por defecto el script usa el `config.json` compartido y los ROIs de `_current_defaults`.
Tambien desactiva overlays y sello de hora en las imagenes de dataset para que el contenido visual sea estable; la hora queda en `manifest.json`.

Para dejarlo capturando cada hora:

```powershell
python pipe_end_detection\capture_pipe_end_dataset.py --loop --interval-minutes 60
```

Para una hora y media:

```powershell
python pipe_end_detection\capture_pipe_end_dataset.py --loop --interval-minutes 90
```

Si quieres que falle cuando una camara no llegue al PTZ pedido:

```powershell
python pipe_end_detection\capture_pipe_end_dataset.py --strict-ptz
```

## Formato esperado de entrada

El script espera dos arboles paralelos:

```text
<images_root>/
  ...
  cam151/
    img_001.jpg
  cam152/
    img_002.jpg

<labels_root>/
  ...
  cam151/
    img_001.txt
  cam152/
    img_002.txt
```

Cada `.txt` debe estar en formato YOLO:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Con valores normalizados en `[0, 1]`.

## Imagenes background

Las imagenes sin tubos de inventario tambien son validas.

Para incluirlas:

- deja la imagen en `images_root`
- no pongas label, o usa un `.txt` vacio
- ejecuta el script con `--allow-missing-labels`

El script generara un `.txt` vacio para esas imagenes en la salida.

## Comando de preparacion

```powershell
python pipe_end_detection\prepare_yolo_dataset.py `
  --images-root pipe_end_detection\annotation_pool\images `
  --labels-root pipe_end_detection\annotation_pool\labels `
  --output-root pipe_end_detection\dataset `
  --allow-missing-labels
```

## Split por defecto

- train: `80%`
- val: `10%`
- test: `10%`

Puedes cambiarlo con:

- `--train-ratio`
- `--val-ratio`
- `--test-ratio`

## Entrenamiento YOLOv11

Ejemplo con Ultralytics:

```powershell
yolo detect train `
  data=pipe_end_detection\data.yaml `
  model=yolo11n.pt `
  imgsz=1280 `
  epochs=100 `
  batch=8
```

## Nota operacional

Este paquete solo prepara y valida el dataset. No reemplaza tu flujo actual del MVP ni del pipeline de medicion.
