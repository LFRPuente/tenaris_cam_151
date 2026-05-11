# Annotation Guide: Pipe End Detection

## Objetivo

Entrenar un detector de extremos visibles de tubos de inventario.

- Tipo: Object Detection
- Formato: YOLO
- Clases: `1`
- Clase unica: `pipe_end`

## Que anotar

Anota una caja ajustada alrededor de la cara cortada visible de cada tubo gris de inventario.

Regla principal:

- anotar solo el extremo
- no anotar el cuerpo completo del tubo
- la caja debe quedar ajustada al tamano real del extremo
- el tamano no debe ser uniforme entre tubos

## Que no anotar

- racks
- soportes
- estructuras metalicas
- marcas de pintura
- etiquetas
- stickers
- tubos sueltos en el suelo que no son inventario
- basura
- herramientas
- vehiculos

## Reglas borderline

Aplicar siempre igual:

- tubo parcialmente tapado: anotar solo si se ve al menos `50%` del extremo
- tubo del fondo de la pila: anotar solo si el inicio individual se distingue claramente
- tubo muy lejano: no anotar si el extremo visible mide menos de `8 px` de ancho
- tubo cortado por borde de imagen: anotar solo si se ve al menos `50%`

## Variedad deseada en el dataset

Incluir:

- distintas condiciones de luz
- sol
- nublado
- mojado
- nieve, si aplica
- distintos angulos de camara
- pilas llenas
- pilas medio llenas
- pilas casi vacias
- aproximadamente `10% a 15%` de imagenes sin tubos de inventario

## Criterio de calidad

Cada annotacion debe cumplir esto:

- la caja rodea solo el extremo
- no incluye partes grandes del cuerpo
- no mezcla dos tubos en una sola caja
- no toca objetos ajenos al extremo si se puede evitar

## Clase YOLO

Usar siempre:

```text
0 pipe_end
```

## Ejemplo de label YOLO

```text
0 0.512500 0.437500 0.026000 0.031000
0 0.544000 0.439000 0.024000 0.029000
```
