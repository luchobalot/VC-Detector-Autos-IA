# Deteccion de autos blancos en video (Vision por Computadora - Parte 1)

## Descripcion del problema y alcance

Dado un video de una calle filmada con camara fija, el sistema detecta los autos,
decide cuales son de color blanco y registra cada aparicion en un archivo CSV
(frame, segundo y evento).

Esta es la primera parte de la evaluacion integradora: preprocesamiento,
deteccion sobre material visual con evidencia (capturas) y registro estructurado.
No incluye entrenamiento de modelos ni otras partes de la evaluacion.

### Como funciona (resumen)

1. Se lee el video frame a frame con OpenCV (que carga en BGR).
2. Cada frame se redimensiona (ancho maximo configurable) y se suaviza levemente.
   Reducir el tamano acelera el procesamiento y el suavizado atenua ruido de compresion.
3. Un modelo YOLO preentrenado (`yolov8n.pt`) detecta solo la clase `car` (id 2 de COCO),
   y el tracker integrado de Ultralytics (`model.track(..., persist=True)`) asigna un ID a cada auto.
4. Para cada auto se recorta su caja, se toma el 60% central (para no medir asfalto ni fondo),
   se pasa a HSV y se calcula el porcentaje de pixeles blancos con `cv2.inRange`
   (saturacion baja, valor alto).
5. Todas las mediciones de un mismo ID se juntan y se decide con el **percentil 20** del
   porcentaje de blanco (configurable en `config.py`, ver abajo).

### Por que se decide por track y no por frame

El porcentaje de blanco de un mismo auto cambia de un frame a otro por reflejos, sombras,
cambios de angulo, desenfoque por movimiento y cajas mal ajustadas. Decidir con un solo
frame hace que un auto gris parezca blanco (o al reves) por una medicion mala.
Resumir varios frames del mismo ID es robusto a esos valores atipicos, y ademas
garantiza un unico evento por auto en lugar de uno por frame.

Se probo primero con la mediana, pero en el video de prueba los autos gris claro subian
a 35-40% de blanco al pasar por una zona con reflejo y la mediana los terminaba
aceptando como blancos. Un percentil bajo (20) exige que el auto sea blanco durante la
mayor parte de su recorrido y no solo en el tramo con reflejo. Se puede cambiar con
`AGGREGATION` ("median", "mean" o "percentile") y `AGGREGATION_PERCENTILE` en `config.py`.
La columna `porcentaje_blanco_medio` de `eventos.csv` guarda el valor agregado con ese metodo.

### Limitaciones conocidas

- **Blanco vs gris claro o plateado:** dependen de los umbrales HSV; hay que calibrarlos.
- **Reflejos y sobreexposicion:** un auto oscuro con un reflejo fuerte puede sumar blanco.
- **Autos tapados:** una caja parcial incluye el objeto que tapa y distorsiona la medicion.
- **Autos lejanos:** son chicos, el recorte tiene pocos pixeles y la medicion es ruidosa.
  Los recortes menores a `MIN_BOX_SIDE_PX` se ignoran.
- **Parametros ajustados a un video de prueba:** el umbral de blanco (23%) y el percentil (20)
  se eligieron con un video de prueba y quedan muy cerca de los valores de algunos autos grises
  claros. Con un video nuevo hay que recalibrar y volver a evaluar.
- **Cambios de ID del tracker:** si el tracker pierde un auto y le da otro ID, se cuenta dos veces.
- **Iluminacion:** de noche o con contraluz los rangos HSV deben recalibrarse.

## Instalacion

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Linux / Mac)
pip install -r requirements.txt
```

La primera ejecucion descarga `yolov8n.pt` automaticamente (requiere internet).

## Ejecucion

Los scripts se ejecutan desde la raiz del proyecto con `python src/<script>.py`.

```
python src/run_pipeline.py --video video/mi_video.mp4
python src/run_pipeline.py --video video/mi_video.mp4 --max-frames 150
python src/run_pipeline.py --video video/mi_video.mp4 --max-seconds 10
python src/run_pipeline.py --video video/mi_video.mp4 --sweep
python src/calibrate_hsv.py --video video/mi_video.mp4 --frame 100
python src/evaluate.py --tolerance 1.5
```

Si no se indica `--video`, se usa el primer video de `video/`.
Otros argumentos de `run_pipeline.py`: `--conf`, `--white-threshold`, `--model`, `--output-dir`.
Todos los valores por defecto estan en `src/config.py`.

## Estructura

| Archivo | Responsabilidad |
|---|---|
| `src/config.py` | Todos los parametros en un solo lugar |
| `src/video_io.py` | Lectura/escritura de video, fps |
| `src/preprocessing.py` | Redimensionado, suavizado y recorte de cajas |
| `src/detector.py` | YOLO + tracking |
| `src/color.py` | HSV, mascara y porcentaje de blanco |
| `src/events.py` | De detecciones por frame a eventos por track ID |
| `src/visualization.py` | Dibujo de cajas y capturas de ejemplo |
| `src/run_pipeline.py` | Script principal |
| `src/calibrate_hsv.py` | Calibracion de rangos HSV |
| `src/evaluate.py` | Comparacion contra anotacion manual |
| `anotaciones/verdad.csv` | Anotacion manual; la actual corresponde al video de prueba, reemplazarla por la del video propio |

## Archivos de salida (carpeta `resultados/`)

- **`eventos.csv`**: una fila por track ID con `track_id, frame_inicio, segundo_inicio,
  frame_fin, segundo_fin, confianza_media, porcentaje_blanco_medio, decision`.
  La decision es `blanco`, `no_blanco` o `insuficiente` (el track tuvo menos de
  `MIN_FRAMES_FOR_DECISION` frames y no se pudo decidir). Los eventos de autos blancos son
  las filas con `decision = blanco`. Se guardan tambien los no blancos para trazabilidad.
- **`detecciones_por_frame.csv`**: una fila por auto por frame con `frame, segundo, track_id,
  confianza, porcentaje_blanco, es_blanco`. Aca `es_blanco` es la medicion de ese frame solo;
  sirve para analizar errores.
- **`video_anotado.mp4`**: cajas verdes (blancos) y rojas (no blancos) con ID y porcentaje.
  El color usa la mediana acumulada hasta ese frame, por eso puede cambiar al principio del track.
- **`capturas/`**: recortes de ejemplo `aceptado_id<ID>_blanco<pct>.jpg` y
  `descartado_id<ID>_blanco<pct>.jpg`. Se borran y regeneran en cada corrida.
- **`barrido_confianza.csv`**: tabla del modo `--sweep` (confianza 0.25, 0.50 y 0.75 con
  cantidad de eventos). Los detalles de cada corrida quedan en `resultados/barrido/`.
- **`calibracion/`**: mascaras de prueba del modo `--headless` de `calibrate_hsv.py`.

## GUIA PARA PROBAR EL SISTEMA

1. **Instalar y verificar.** Seguir la instalacion y correr
   `python -c "import ultralytics, cv2; print(ultralytics.__version__, cv2.__version__)"`.
2. **Prueba rapida.** Poner un video corto en `video/` (por ejemplo de Pexels o Pixabay) y correr
   `python src/run_pipeline.py --video video/mi_video.mp4 --max-frames 150`.
   Comprobar que se generan los archivos en `resultados/`.
3. **Revisar el video anotado.** Abrir `resultados/video_anotado.mp4` y verificar a ojo que las
   cajas esten bien ubicadas sobre los autos.
4. **Calibrar HSV.** Correr `python src/calibrate_hsv.py --video video/mi_video.mp4 --frame N`
   con un frame donde haya autos blancos y otros claros. Ajustar los trackbars hasta que la
   mascara resalte solo los blancos, cerrar la ventana y copiar los valores impresos a
   `HSV_LOWER` y `HSV_UPPER` en `config.py`. Sin pantalla, usar `--headless` y elegir entre
   las imagenes de `resultados/calibracion/`.
5. **Revisar capturas.** Mirar `resultados/capturas/` y buscar errores tipicos: autos grises o
   plateados aceptados, autos blancos descartados, reflejos.
6. **Evaluar.** Anotar a mano en `anotaciones/verdad.csv` el segundo en que pasa cada auto
   blanco (una fila por auto, columna `segundo`) y correr `python src/evaluate.py`.
   Un auto real cuenta como detectado si su segundo cae dentro del intervalo
   `[segundo_inicio, segundo_fin]` de un evento blanco, mas o menos la tolerancia.
7. **Barrer umbrales de confianza.** Correr `python src/run_pipeline.py --video ... --sweep`
   y comparar la tabla `resultados/barrido_confianza.csv`.
8. **Interpretar.**
   - Muchos falsos positivos: subir `WHITE_PERCENT_THRESHOLD` o el umbral de confianza,
     o achicar el rango de saturacion maxima (S max).
   - Se pierden autos: bajar esos umbrales o `MIN_FRAMES_FOR_DECISION`.
   - Muchos tracks `insuficiente`: el auto aparece pocos frames o el tracker lo pierde.
   - Si cambian la camara, la hora o la iluminacion, repetir la calibracion y la evaluacion
     con videos propios para confirmar que los parametros siguen sirviendo.

## Como citar fuentes de videos de terceros

Si usas videos de Pexels, Pixabay u otros sitios, anota para cada uno: **autor**, **sitio**,
**link** y **licencia**. Esa informacion debe aparecer en el informe de la entrega.
Los videos no se incluyen en este repositorio.
