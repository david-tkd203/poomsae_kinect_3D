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
    │   └── csv_backend.py           # Lectura de CSV de landmarks precomputados
    │
    ├── segmentation/
    │   └── segmenter.py             # Segmentación de movimientos (onsets/offsets)
    │
    ├── features/
    │   ├── feature_extractor.py     # Cálculo de features cinemáticas
    │   └── angles.py                # Utilidades de cálculo angular
    │
    ├── utils/
    │   └── geometry.py              # Funciones geométricas generales
    │
    ├── rules/
    │   └── deduction_rules.py       # Reglas de deducción (−0.1 / −0.3)
    │
    ├── model/
    │   ├── dataset_builder.py       # Construcción del dataset de entrenamiento
    │   ├── train.py                 # Entrenamiento de modelo ML clásico
    │   └── infer.py                 # Inferencia/predicción en nuevas ejecuciones
    │
    ├── annotations/                 # Etiquetas manuales por movimiento
    ├── configs/                     # Configuración de paths, parámetros, poomsae
    ├── notebooks/                   # Análisis exploratorios, prototipos
    └── ui/                          # Tkinter/Django u otra interfaz (si aplica)
