# Pipe End YOLO Project

Proyecto limpio para anotar y entrenar deteccion `pipe_end`.

## Anotacion

Imagenes:

```text
annotation_pool/images/cam151
annotation_pool/images/cam152
```

Labels YOLO:

```text
annotation_pool/labels/cam151
annotation_pool/labels/cam152
```

Clase unica:

```text
0 pipe_end
```

## App local de anotacion

```powershell
python scripts\annotate_app.py
```

Uso:

- arrastra con el mouse para dibujar una caja
- click sobre una caja para seleccionarla
- `Delete` borra la caja seleccionada
- `Ctrl+S` guarda el label YOLO
- flechas izquierda/derecha cambian de imagen
- filtro lateral permite ver `cam151`, `cam152`, pendientes o anotadas
- `Bad warp` marca una imagen para excluirla del dataset
- `Load AI boxes` carga predicciones YOLO para corregirlas manualmente

## Preparar dataset YOLO

```powershell
python scripts\prepare_yolo_dataset.py `
  --images-root annotation_pool\images `
  --labels-root annotation_pool\labels `
  --output-root dataset `
  --allow-missing-labels
```

Las imagenes marcadas como `Bad warp` se excluyen automaticamente.

## Active learning

1. Anota manualmente al menos `15-25` imagenes buenas.
2. Entrena primer modelo:

```powershell
python scripts\train_yolo.py
```

3. Genera predicciones sobre imagenes no anotadas:

```powershell
python scripts\predict_unlabeled.py --weights runs\pipe_end\train\weights\best.pt
```

4. En la app, abre una imagen con badge `AI`, usa `Load AI boxes`, corrige las cajas y guarda.

## Entrenar con Ultralytics

```powershell
yolo detect train data=data.yaml model=yolo11n.pt imgsz=1280 epochs=100 batch=8
```
