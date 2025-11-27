# Prototipo de Sistema de Evaluación Automatizada de Exactitud en Poomsae (Taekwondo WT)

> Tesis de Licenciatura en Ingeniería Civil en Informática y Telecomunicaciones  
> Universidad Finis Terrae – 2025  
> Autor: David Ñanculeo

Este repositorio contiene el prototipo de un sistema de **evaluación automatizada de la nota de exactitud en Poomsae reconocido de Taekwondo WT**, basado en **visión por computador**, **estimación de pose** y un **modelo de Machine Learning interpretable** (árbol / bosque de decisión).

El sistema:

- Captura ejecuciones de Poomsae con **múltiples cámaras ópticas** y una **Kinect** como apoyo de profundidad.
- Extrae **landmarks esqueléticos** (33 puntos) por frame usando **MediaPipe Pose**.
- Segmenta la ejecución en **unidades técnicas** (movimientos individuales).
- Calcula **métricas cinemáticas** (ángulos, alineación, profundidad de posturas, estabilidad, giros, altura de patadas, etc.).
- Clasifica cada movimiento como **Correcto**, **Error Leve (−0.1)** o **Error Grave (−0.3)** según la lógica del reglamento WT.
- Agrega las deducciones para obtener la **nota de exactitud** (máx. 4.0, mín. 1.5) y genera un **reporte auditable en Excel** con marcas de tiempo y métricas asociadas. :contentReference[oaicite:0]{index=0}


---

## 1. Motivación de la tesis

En la modalidad de **Poomsae reconocido**, la evaluación de la exactitud depende casi por completo del **criterio subjetivo de los jueces**, aunque el reglamento WT define de manera explícita:

- La nota máxima de exactitud (**4.0 puntos**).
- Las deducciones normadas:
  - **−0.1** por error leve.
  - **−0.3** por error grave.
- La nota mínima de exactitud (**1.5 puntos**).

Esto genera varios problemas:

1. **Variabilidad inter e intra-juez** en la aplicación de las deducciones.
2. **Falta de transparencia y trazabilidad** en las decisiones de puntuación.
3. Ausencia de un mecanismo objetivo que **vincule la técnica observada** (ángulos, posturas, giros, equilibrio) con las deducciones numéricas del reglamento.

La tesis propone un **pipeline determinista y auditable** que:

- Convierte la ejecución de Poomsae en **datos cinemáticos medibles**.
- Aplica reglas explícitas y un modelo ML interpretable para **mapear métricas físicas → deducciones (−0.1 / −0.3)**.
- Genera **informes reutilizables y comparables** para atletas, entrenadores y jueces.

---

## 2. Objetivos del proyecto

### 2.1 Objetivo general

> Diseñar y construir un prototipo para la evaluación en Poomsae utilizando **secuencias de video en tiempo real** y un **modelo de Machine Learning de árbol de decisiones** como apoyo a la práctica del Taekwondo WT.

### 2.2 Objetivos específicos

- Desarrollar un **sistema de captura en tiempo real** basado en **Kinect** y **cámaras ópticas**.
- Implementar un **pipeline de procesamiento** para:
  - Estimación de pose (landmarks).
  - Fusión de datos de profundidad (Kinect).
  - Extracción de métricas cinemáticas.
- Entrenar y optimizar un **clasificador de movimientos** (correcto / error leve / error grave) alineado con las reglas WT.
- Desarrollar un **sistema de reporte auditable** (nota final, detalle de errores, marcas de tiempo).
- Validar el prototipo contra **jueces expertos**, midiendo:
  - **ICC** (concordancia).
  - **MAE** (diferencia de nota).
  - **F1 macro** (detección de errores).
  - **Latencia** del sistema.

---

## 3. Arquitectura general del sistema

El sistema sigue el flujo:

1. **Captura**  
   Multivista + profundidad:
   - ≥ 4 cámaras ópticas (30–60 fps).
   - 1 sensor **Kinect** (RGB-D).
   - Sincronización por timestamp / eventos.

2. **Estimación de pose (landmarks)**  
   - Uso de **MediaPipe Pose** para extraer 33 puntos esqueléticos por frame.
   - Coordinadas **normalizadas** para robustez inter-sujeto.

3. **Segmentación temporal**  
   - Cálculo de **energía por frame** en efectores (muñecas y tobillos).
   - Umbrales adaptativos (mediana + MAD) para detectar **onsets/offsets**.
   - Cada segmento ≈ un **movimiento técnico** (unidad de Poomsae).

4. **Extracción de métricas cinemáticas**  
   Ejemplos:
   - Ángulos de rodilla, codo, cadera, hombro.
   - Profundidad de postura (distancia pie-pie, flexión de rodilla).
   - Orientación corporal (giros 90°/180°).
   - Altura y amplitud de patadas.
   - Estabilidad y desplazamiento del centro de masa.

5. **Clasificación y scoring reglamentario**  
   - Reglas deterministas + modelo ML clásico (Decision Tree / Random Forest / SVM).
   - Asignación de etiquetas:
     - `OK`
     - `ERROR_LEVE` → −0.1
     - `ERROR_GRAVE` → −0.3
   - **Nota exactitud** = `4.0 − Σ(deducciones)` con mínimo 1.5.

6. **Reporte auditable**  
   - Generación de archivo **Excel (.xlsx)** con:
     - Nota de exactitud final.
     - Tabla de movimientos (por orden) y deducciones.
     - Marcas de tiempo por movimiento y por error.
     - Métricas cinemáticas de respaldo.
   - Registro en **base de datos** (SQLite en desarrollo, MySQL en piloto).

---

## 4. Estructura del repositorio

> Nota: adapta los nombres de carpetas si tu estructura difiere; aquí se sigue el esquema propuesto en la tesis.

```text
poomsae_accuracy/
├── README.md
├── requirements.txt
├── environment.yml                  # (opcional) entorno conda
└── src/
    ├── main_offline.py              # Pipeline completo desde archivos de video
    ├── main_realtime.py             # Pipeline en tiempo real con cámaras/Kinect
    │
    ├── video_get.py                 # Captura / descarga básica de videos
    ├── dataio/
    │   ├── video_downloader.py      # Descarga y organización de videos
    │   └── report_generator.py      # Generación de reportes .xlsx
    │
    ├── preprocess/
    │   └── bg_remove.py             # (Opcional) eliminación de fondo
    │
    ├── pose/
    │   ├── mediapipe_backend.py     # Wrapper de MediaPipe Pose
   # Prototipo de Sistema de Evaluación Automatizada de Exactitud en Poomsae (Taekwondo WT)

   > Tesis de Licenciatura en Ingeniería Civil en Informática y Telecomunicaciones  
   > Universidad Finis Terrae – 2025  
   > Autor: David Ñanculeo

   Este repositorio contiene el código del prototipo de evaluación de la nota de exactitud en Poomsae (Taekwondo WT). El README que sigue está actualizado con el árbol real de `src/` presente en este workspace y una descripción práctica y detallada de cada archivo Python (propósito, responsabilidades y puntos clave de implementación).

   Contenido principal:
   - Captura con Kinect RGB-D y cámaras ópticas.
   - Extracción de landmarks con MediaPipe Pose (33 puntos).
   - Segmentación temporal de movimientos técnicos.
   - Extracción de métricas cinemáticas y scoring reglamentario.
   - Generación de reportes y utilidades para entrenamiento e inferencia.

   ---

   ## Árbol actual (relevante) — `src/`

   La estructura real detectada en el repositorio (lista de archivos Python bajo `src/`):

   ```
   src/
   ├── config.py
   ├── perf_tuning.py
   ├── main_kinect.py
   ├── recording/
   │   └── session_recorder.py
   ├── kinect/
   │   └── kinect_capture.py
   ├── pose/
   │   └── mp_pose_wrapper.py
   ├── viz/
   │   ├── kinect_3d_window.py
   │   └── report_window.py
   ├── offline/
   │   └── offline_pipeline.py
   ├── segmentation/
   │   ├── segmenter.py
   │   └── move_capture.py
   ├── tools/
   │   ├── extract_landmarks.py
   │   ├── view_kinect_npz.py
   │   └── score_pal_yang.py
   ├── dataio/
   │   ├── report_generator.py
   │   └── score_exporter.py
   ├── features/
   │   ├── feature_extractor.py
   │   └── angles.py
   ├── ml/
   │   ├── train.py
   │   ├── stance_classifier.py
   │   └── build_stance_dataset.py
   ├── utils/
   │   ├── smoothing.py
   │   └── performance.py
   └── __init__.py
   ```

   > Nota: el árbol anterior refleja los módulos y ficheros Python detectados automáticamente. Si tu entorno local tiene archivos adicionales (por ejemplo `captures/`, `data/` o notebooks), no aparecen aquí porque son datos o artefactos de ejecución.

   ---

   ## Descripción detallada por archivo (qué hace y puntos clave)

   Abajo describo cada archivo Python presente en `src/`. El objetivo es que cualquier desarrollador (o tú mismo) entienda rápidamente la responsabilidad de cada módulo y dónde mirar cuando necesite cambiar comportamiento.

   - `config.py` — Configuraciones globales
      - Contiene rutas por defecto, parámetros experimentales (fps, resoluciones), y constantes reutilizadas por distintos módulos.
      - Punto clave: centraliza paths (por ejemplo `captures/`, `models/`) y parámetros del segmentador y extractor de features.

   - `perf_tuning.py` — Utilidades de rendimiento
      - Código para pruebas/perfiles y experimento de latencia. Incluye benchs simples y wrappers para medir tiempos de extracción y serialization.

   - `main_kinect.py` — Entrada para captura con Kinect y grabación
      - Script de alto nivel que inicializa `kinect_capture`, la UI (opcional) y el `session_recorder` para grabar `.npz` y video mp4.
      - Punto clave: maneja la creación de sesiones (`captures/session_<timestamp>/`) y coordina la captura de color + nubes 3D.

   - `recording/session_recorder.py` — Gestión de grabación y flush
      - `RecordingSession` y `Kinect3DRecorder` (o clases similares): responsable de escribir `kinect_3d_data.npz`, sincronizar frames, y almacenar video color (`kinect_color.mp4`).
      - Detalle: cuida buffering, estimación de FPS y cierre seguro de writers.

   - `kinect/kinect_capture.py` — Wrapper de Kinect (pykinect2)
      - Abstrae acceso a PyKinectRuntime/PyKinectV2: obtiene color frames, depth, body joints y la conversión a nubes de puntos 3D.
      - Funciones importantes: `update_frames()`, `get_joint_positions()`, `get_body_point_cloud()`.
      - Nota: contiene transformaciones de coordenadas (Kinect → sistema del proyecto) y manejo de cuerpos no rastreados.

   - `pose/mp_pose_wrapper.py` — Wrapper de MediaPipe Pose (estimator)
      - Punto central para la extracción de landmarks 2D/3D por frame (33 puntos). Implementa inicialización (modelo), inferencia por frame y utilidades de postprocesado (normalización, visibilidad, filtrado temporal).
      - Afecta directamente la calidad de los `*_landmarks.csv` o arrays `.npz` que luego usa el segmentador.

   - `viz/kinect_3d_window.py` — Visualizador 3D en tiempo real
      - UI basada en PyQt5 y `pyqtgraph.opengl` que representa la nube de puntos y esqueleto 3D. Incluye control de cámara, reproducción y marcadores de movimientos.
      - Útil para validar que la nube y joints están alineados correctamente con el color stream.

   - `viz/report_window.py` — Ventana de reporte / revisión
      - Interfaz para abrir XSLX / JSON de scoring y mostrar el resumen gráfico (tabla de movimientos, deducciones por movimiento y reproductor de video sincronizado).

   - `offline/offline_pipeline.py` — Orquestador offline
      - Toma una captura (carpeta con `kinect_3d_data.npz` y `kinect_color.mp4`) y ejecuta paso a paso:
         1. Extracción de landmarks (si no existen).
         2. Segmentación en movimientos (`segmenter`).
         3. Extracción de features por movimiento.
         4. Scoring con reglas y/o modelo ML.
         5. Export a `report.xlsx` y/o JSON.
      - Observación: la pipeline asume la presencia de archivos `*_moves.json` en la carpeta `moves/` para algunos flujos; si faltan, ejecuta el segmentador.

   - `segmentation/segmenter.py` — Algoritmos de segmentación
      - Implementa energía por frame, suavizado y detección de picos/onsets. Exporta funciones para obtener intervals (start,end) y `moves.json` con metadata (timestamps, frames).
      - Parámetros clave: ventana de suavizado, umbral por mediana+MAD, ruido mínimo de movimiento.

   - `segmentation/move_capture.py` — Captura y heurísticas de movimiento
      - Contiene lógica específica para Poomsae: identificación de la extremidad activa (manos/pies), heurísticas para merges y separación de sub-movimientos, y utilidades para escribir `*_moves.json` consumibles por scoring.

   - `tools/extract_landmarks.py` — Herramienta CLI para extracción de landmarks
      - Script utilitario que recorre videos o capturas, ejecuta el `mp_pose_wrapper` y escribe CSV/NPZ con landmarks por frame.
      - Opciones comunes: especificar carpeta de entrada, salida, y toggle para sobrescribir o reutilizar resultados existentes.

   - `tools/view_kinect_npz.py` — Viewer de `.npz` Kinect
      - Programa para inspeccionar grabaciones 3D guardadas. Dibuja nube de puntos y esqueleto y permite avanzar frame a frame.
      - Muy útil para debugging y validación visual de la sincronía color→nube→landmarks.

   - `tools/score_pal_yang.py` — Script de scoring específico (ej. Poomsae Pal-Yang)
      - Implementa el cálculo de la nota final y heurísticas de ajuste relacionadas con patrones de la forma Pal-Yang.

   - `dataio/report_generator.py` — Generador de reportes `.xlsx` / tablas
      - Funciones para crear hojas de cálculo con la tabla de movimientos, deducciones, métricas y resumen final.
      - Entrada típica: dataframe con movimientos + columnas `start_frame`, `end_frame`, `deduction`, `reason`.

   - `dataio/score_exporter.py` — Exportador y helpers de I/O
      - Funciones para serializar resultados a JSON/CSV/NPZ y empaquetar los artefactos de una sesión para revisión o subida.

   - `features/feature_extractor.py` — Cálculo de métricas por movimiento
      - Extrae features por segmento (ángulos de articulaciones, desplazamientos, alturas de patada, estabilidad temporal, RMS de error respecto a template).
      - Estas features son las entradas del clasificador/árbol.

   - `features/angles.py` — Utilidades geométricas y angulares
      - Funciones para calcular ángulos entre vectores, unwrap de ángulos, y conversión entre sistemas de coordenadas.

   - `ml/train.py` — Entrenamiento de modelos ML clásicos
      - Entrena RandomForest/DecisionTree, evalúa con CV y guarda el modelo serializado (`.pkl`). Contiene rutinas para balanceo y generación de folds.

   - `ml/stance_classifier.py` — Clasificador de posturas (ejemplo)
      - Implementa la API de inferencia para un clasificador de stances (posturas) — carga modelo, `predict_proba` y utilidades de post-proc.

   - `ml/build_stance_dataset.py` — Generación de dataset para ML
      - Script que lee features extraídas y labels manuales para producir `X,y` listos para entrenamiento.

   - `utils/smoothing.py` — Filtros y suavizado temporal
      - Implementa moving-average, gaussian smoothing y funciones para eliminar jitter de landmarks.

   - `utils/performance.py` — Medidas y utilidades para performance
      - Timers, contadores y registros legibles para medir throughput y latencia en pipelines.

   ---

   ## Comandos útiles y flujo típico de uso

   1. Activar el virtualenv (PowerShell):

   ```powershell
   .\\.venv\\Scripts\\Activate.ps1
   ```

   2. Extraer landmarks desde un video o carpeta de videos (ejemplo):

   ```powershell
   python -m src.tools.extract_landmarks --input videos/ --output captures/ --overwrite
   ```

   3. Ejecutar la pipeline offline sobre una captura:

   ```powershell
   python -m src.offline.offline_pipeline --capture captures/session_YYYYMMDD_HHMMSS
   ```

   4. Ver una captura Kinect `.npz`:

   ```powershell
   python -m src.tools.view_kinect_npz captures\\session_YYYYMMDD_HHMMSS\\kinect_3d_data.npz
   ```

   5. Generar reporte Excel desde los resultados guardados:

   ```powershell
   python -m src.dataio.report_generator --input results/session_xxx --output report.xlsx
   ```

   ---

   ## Siguientes recomendaciones

   - Versionar sólo el código: añade `captures/` y `data/` a `.gitignore` (ya se propuso). Para eliminar artefactos ya trackeados, usar `git rm -r --cached captures data` y commitear.
   - Generar documentación automática (Sphinx) a partir de los docstrings de `src/`.
   - Añadir tests unitarios básicos para funciones puras (por ejemplo `features/angles.py`, `utils/smoothing.py`) y tests de integración para `offline_pipeline` usando un conjunto de muestras pequeñas.
   - Añadir un `requirements.txt` (o `pyproject.toml`) con versiones pinneadas para reproducibilidad.

   ---

   Si quieres, aplico ahora el parche para commitear el README actualizado (añadir, commit message sugerido) o continúo documentando cada función pública dentro de cada archivo Python (añadir docstrings). ¿Qué prefieres que haga a continuación?
