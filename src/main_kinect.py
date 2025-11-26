# src/main_kinect.py
from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt5 import QtWidgets

from .viz import Kinect3DWindow
from .segmentation.move_capture import capture_moves_with_spec, CaptureResult

# Opcional: reporte GT vs predicción (tu módulo antiguo)
try:
    from .dataio.report_generator import (
        load_labels,
        load_predictions,
        match_and_score,
        export_excel,
    )
except Exception:
    load_labels = load_predictions = match_and_score = export_excel = None  # type: ignore

# NUEVO: usamos directamente tu score_pal_yang.py
try:
    from .tools.score_pal_yang import score_many_to_excel as palyang_score_many_to_excel
except Exception:
    palyang_score_many_to_excel = None  # type: ignore

# =====================================================================
# RUTAS BASE DEL PROYECTO
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

LANDMARKS_DIR = DATA_DIR / "landmarks" / "8yang"
PREVIEWS_DIR = DATA_DIR / "previews"
MOVES_DIR = DATA_DIR / "moves_ml"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
DEFAULT_SPEC = CONFIG_DIR / "patterns" / "8yang_spec.json"
DEFAULT_POSE_SPEC = CONFIG_DIR / "patterns" / "pose_spec.json"

DEFAULT_ML_MODEL = DATA_DIR / "models" / "stance_classifier.joblib"


# =====================================================================
# UTILIDADES DE SCORE (match_score medio, para log rápido)
# =====================================================================

def _score_to_label(score: float) -> str:
    """
    Mapea match_score ∈ [0,1] a etiquetas cualitativas para logs rápidos.

      - score >= 0.80  -> "correcto"
      - 0.50–0.80      -> "error_leve"
      - < 0.50         -> "error_grave"
    """
    if score >= 0.80:
        return "correcto"
    if score >= 0.50:
        return "error_leve"
    return "error_grave"


def _compute_pal_yang_score(result: CaptureResult) -> Dict[str, float]:
    """
    Promedio de match_score del pipeline (0–1) y versión 0–100.
    OJO: La nota oficial WT está en el Excel generado por score_pal_yang.
    """
    if not result.moves:
        return {"avg_match_score": 0.0, "pal_yang_score": 0.0}

    total = 0.0
    n = 0
    for m in result.moves:
        try:
            total += float(m.match_score)
            n += 1
        except Exception:
            pass

    avg_match = (total / n) if n > 0 else 0.0
    pal_yang = round(avg_match * 100.0, 2)
    return {
        "avg_match_score": float(avg_match),
        "pal_yang_score": float(pal_yang),
    }


# =====================================================================
# RESOLVER RUTAS EN MODO OFFLINE
# =====================================================================

def _video_id_from_sample(sample_id: str) -> str:
    """
    ID lógico de video (para reportes).
    Por defecto usamos el propio sample_id (ej: '8yang_001').
    """
    return sample_id


def _build_paths_from_sample(
    sample_id: str,
    csv_path_cli: Optional[str],
    video_path_cli: Optional[str],
) -> tuple[Path, Optional[Path]]:
    """
    Resuelve rutas a CSV de landmarks y preview de video para un `sample_id`.

    Convención actual:
      - CSV:    data/landmarks/8yang/<sample_id>.csv
                (ej: 8yang_001 -> data/landmarks/8yang/8yang_001.csv)
      - Video:  data/previews/8yang_<sample_id>_preview.mp4
                (ej: 8yang_001 -> data/previews/8yang_8yang_001_preview.mp4)
    """
    # ----- CSV -----
    if csv_path_cli:
        csv_path = Path(csv_path_cli)
    else:
        csv_path = LANDMARKS_DIR / f"{sample_id}.csv"

    if not csv_path.exists():
        raise SystemExit(
            f"[OFFLINE] No se encontró CSV de landmarks para sample_id={sample_id}\n"
            f"  Buscado en: {csv_path}"
        )

    # ----- VIDEO (preview) -----
    video_path: Optional[Path] = None
    if video_path_cli:
        vp = Path(video_path_cli)
        if vp.exists():
            video_path = vp
        else:
            print(f"[OFFLINE] ⚠️ Video CLI no existe: {vp}, continuo sin video.")
    else:
        poomsae_name = sample_id.split("_")[0]  # ej: "8yang"
        candidate = PREVIEWS_DIR / f"{poomsae_name}_{sample_id}_preview.mp4"
        if candidate.exists():
            video_path = candidate
        else:
            fallback = PREVIEWS_DIR / f"{sample_id}_preview.mp4"
            if fallback.exists():
                video_path = fallback
            else:
                print(
                    "[OFFLINE] ⚠️ No se encontró preview de video para "
                    f"{sample_id}. Continuo solo con landmarks."
                )

    return csv_path, video_path


# =====================================================================
# JSON PARA REPORTES (GT vs pred) — mantengo summary/segments
# =====================================================================

def _augment_json_for_report(result: CaptureResult, video_id: str) -> Dict[str, Any]:
    """
    Convierte CaptureResult a dict y añade:

      - 'segments': lista {start_s, end_s, label, ...}
        compatible con report_generator._extract_segments_from_json
      - 'summary': {'pal_yang_score', 'avg_match_score', 'n_segments'}
      - 'video_id': ID lógico del video usado en los reportes.
    """
    base: Dict[str, Any] = json.loads(result.to_json())
    base["video_id"] = video_id

    segments: List[Dict[str, Any]] = []
    for m in result.moves:
        score = float(getattr(m, "match_score", 0.0) or 0.0)
        label = _score_to_label(score)
        segments.append(
            {
                "start_s": float(getattr(m, "t_start", 0.0)),
                "end_s": float(getattr(m, "t_end", 0.0)),
                "label": label,
                "move_idx": int(getattr(m, "idx", -1)),
                "stance_pred": getattr(m, "stance_pred", ""),
                "kick_pred": getattr(m, "kick_pred", ""),
                "rotation_bucket": getattr(m, "rotation_bucket", ""),
                "match_score": score,
            }
        )

    base["segments"] = segments

    score_info = _compute_pal_yang_score(result)
    base["summary"] = {
        "pal_yang_score": score_info["pal_yang_score"],      # 0–100
        "avg_match_score": score_info["avg_match_score"],    # 0–1
        "n_segments": len(result.moves),
    }

    return base


# =====================================================================
# MODOS DE EJECUCIÓN
# =====================================================================

def run_gui() -> None:
    """
    Modo GUI en vivo:
      - Abre la ventana Kinect 3D.
      - La lógica de grabar video + landmarks y luego llamar al
        pipeline de análisis se implementa dentro de Kinect3DWindow.
    """
    app = QtWidgets.QApplication(sys.argv)
    w = Kinect3DWindow()
    w.resize(960, 720)
    w.show()
    sys.exit(app.exec_())


def run_offline(args: argparse.Namespace) -> None:
    """
    Modo offline:
      - Toma un sample_id (ej: 8yang_001) o un CSV directo.
      - Carga landmarks CSV + preview de video (si existe).
      - Corre capture_moves_with_spec (segmentación + postura + patada).
      - Guarda JSON en data/moves_ml/<sample_id>_moves.json.
      - Llama a score_pal_yang.score_many_to_excel para generar:
            reports/<sample_id>_score_final.xlsx
        con todas tus reglas/afinamientos originales.
    """
    if not args.sample_id and not args.csv:
        raise SystemExit(
            "[OFFLINE] Debes indicar --sample-id 8yang_001 o --csv ruta/al/archivo.csv"
        )

    # Resolver sample_id
    if args.sample_id:
        sample_id = args.sample_id
    else:
        sample_id = Path(args.csv).stem

    video_id = _video_id_from_sample(sample_id)

    csv_path, video_path = _build_paths_from_sample(
        sample_id=sample_id,
        csv_path_cli=args.csv,
        video_path_cli=args.video,
    )

    # Configs
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    spec_path = Path(args.spec) if args.spec else DEFAULT_SPEC
    pose_spec_path = Path(args.pose_spec) if args.pose_spec else DEFAULT_POSE_SPEC

    if not config_path.exists():
        raise SystemExit(f"[OFFLINE] No existe config YAML: {config_path}")
    if not spec_path.exists():
        raise SystemExit(f"[OFFLINE] No existe 8yang_spec.json: {spec_path}")
    if not pose_spec_path.exists():
        raise SystemExit(f"[OFFLINE] No existe pose_spec.json: {pose_spec_path}")

    # Modelo ML de posturas (opcional)
    use_ml_classifier = bool(args.ml)
    ml_model_path: Optional[Path] = None
    if use_ml_classifier:
        ml_model_path = Path(args.ml)
        if not ml_model_path.exists():
            print(f"[OFFLINE] ⚠️ Modelo ML no encontrado en {ml_model_path}, continuo sin ML.")
            use_ml_classifier = False

    print("==============================================================")
    print(f"[OFFLINE] Sample ID        : {sample_id}")
    print(f"[OFFLINE] Video ID (report): {video_id}")
    print(f"[OFFLINE] CSV landmarks    : {csv_path}")
    print(f"[OFFLINE] Video preview    : {video_path if video_path else '(sin video)'}")
    print(f"[OFFLINE] Config YAML      : {config_path}")
    print(f"[OFFLINE] 8yang_spec.json  : {spec_path}")
    print(f"[OFFLINE] pose_spec.json   : {pose_spec_path}")
    print(f"[OFFLINE] ML classifier    : {'ON' if use_ml_classifier else 'OFF'}")
    print("==============================================================")

    # --------------------- SEGMENTACIÓN + CLASIFICACIÓN --------------------
    result: CaptureResult = capture_moves_with_spec(
        csv_path=csv_path,
        video_path=video_path,
        config_path=config_path,
        spec_path=spec_path,
        pose_spec_path=pose_spec_path,
        use_ml_classifier=use_ml_classifier,
        ml_model_path=ml_model_path,
    )

    print(f"[OFFLINE] Movimientos detectados: {len(result.moves)}")

    if not result.moves:
        print("[OFFLINE] ⚠️ No se detectaron movimientos, nada que guardar.")
        return

    # --------------------- LOG RÁPIDO MATCH_SCORE PROMEDIO -----------------
    score_info = _compute_pal_yang_score(result)
    print(
        f"[OFFLINE] Match medio (score interno) * 100: "
        f"{score_info['pal_yang_score']:.2f} "
        f"(avg_match_score={score_info['avg_match_score']:.3f})"
    )
    print("          Nota WT oficial = 'exactitud_final' en hoja 'resumen' del Excel.")

    # --------------------- GUARDAR JSON DE MOVIMIENTOS ---------------------
    MOVES_DIR.mkdir(parents=True, exist_ok=True)
    out_json = MOVES_DIR / f"{sample_id}_moves.json"

    full_dict = _augment_json_for_report(result, video_id=video_id)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_dict, f, ensure_ascii=False, indent=2)

    print(f"[OFFLINE] JSON de movimientos + segments + summary -> {out_json}")

    # --------------------- EXCEL CON score_pal_yang.py ---------------------
    if palyang_score_many_to_excel is not None:
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            alias = sample_id.split("_")[0]  # ej: "8yang"
            out_xlsx = REPORTS_DIR / f"{sample_id}_score_final.xlsx"

            palyang_score_many_to_excel(
                moves_jsons=[out_json],
                spec_json=spec_path,
                out_xlsx=out_xlsx,
                landmarks_root=DATA_DIR / "landmarks",
                alias=alias,
                subset=None,                 # sin subcarpeta (usa data/landmarks/8yang/...)
                pose_spec=pose_spec_path,
                penalty_leve=0.1,
                penalty_grave=0.3,
                clamp_min=1.5,
                restarts_map=None,
            )
            print(f"[OFFLINE] ✅ Score Pal Yang XLSX -> {out_xlsx}")
        except Exception as e:
            print(f"[OFFLINE] ⚠️ Error generando Excel Pal Yang: {e}")
    else:
        print("[OFFLINE] ⚠️ score_pal_yang.py no está disponible, no se genera XLSX.")

    # --------------------- OPCIONAL: REPORTE GT vs PRED --------------------
    if args.labels and args.report_xlsx and load_labels and load_predictions:
        labels_csv = Path(args.labels)
        out_xlsx_gt = Path(args.report_xlsx)
        preds_dir = MOVES_DIR  # Carpeta donde acabamos de dejar los JSON

        if not labels_csv.exists():
            print(f"[OFFLINE] ⚠️ labels CSV no existe: {labels_csv}, omito reporte GT.")
            return

        print(f"[OFFLINE] Generando reporte de acierto vs GT:")
        print(f"          labels_csv = {labels_csv}")
        print(f"          preds_dir  = {preds_dir}")
        print(f"          out_xlsx   = {out_xlsx_gt}")

        gt_df = load_labels(labels_csv)
        pr_df = load_predictions(preds_dir)

        vids_gt = set(gt_df["video_id"])
        vids_pr = set(pr_df["video_id"])
        vids_common = vids_gt & vids_pr
        if not vids_common:
            print("[OFFLINE] ⚠️ No hay intersección de videos entre labels y predicciones.")
            return

        gt_df = gt_df[gt_df["video_id"].isin(vids_common)].reset_index(drop=True)
        pr_df = pr_df[pr_df["video_id"].isin(vids_common)].reset_index(drop=True)

        details, metrics = match_and_score(gt_df, pr_df)
        out = export_excel(out_xlsx_gt, metrics, details)
        print(f"[OFFLINE] ✅ Reporte XLSX GT vs pred -> {out}")


# =====================================================================
# ARGPARSE / ENTRYPOINT
# =====================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Poomsae Kinect 3D - captura en vivo y análisis offline (8yang)."
    )

    ap.add_argument(
        "--mode",
        choices=["gui", "offline"],
        default="gui",
        help="Modo de ejecución: 'gui' (ventana Kinect 3D) o 'offline' (procesar CSV).",
    )

    # ---- Parámetros para modo offline ----
    ap.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="ID del sample a procesar (ej: 8yang_001).",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Ruta directa a CSV de landmarks (omite sample-id).",
    )
    ap.add_argument(
        "--video",
        type=str,
        default=None,
        help="Ruta directa a video/preview (omite búsqueda automática).",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Config YAML (por defecto: {DEFAULT_CONFIG})",
    )
    ap.add_argument(
        "--spec",
        type=str,
        default=None,
        help=f"JSON de especificación de poomsae (por defecto: {DEFAULT_SPEC})",
    )
    ap.add_argument(
        "--pose-spec",
        type=str,
        default=None,
        help=f"JSON de especificación de posturas/patadas (por defecto: {DEFAULT_POSE_SPEC})",
    )
    ap.add_argument(
        "--ml",
        type=str,
        default=None,
        help="Ruta a modelo ML de posturas (joblib/pkl). Si se entrega, activa PoseClassifier.use_ml.",
    )

    # Opcionales para reporte GT vs pred automático
    ap.add_argument(
        "--labels",
        type=str,
        default=None,
        help="CSV de etiquetas GT para generar reporte (opcional).",
    )
    ap.add_argument(
        "--report-xlsx",
        type=str,
        default=None,
        help="Ruta XLSX de salida para el reporte GT vs pred (opcional).",
    )

    return ap


def main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    ap = build_argparser()
    args = ap.parse_args(argv)

    if args.mode == "gui":
        run_gui()
    elif args.mode == "offline":
        run_offline(args)
    else:
        raise SystemExit(f"Modo desconocido: {args.mode}")


if __name__ == "__main__":
    main()
