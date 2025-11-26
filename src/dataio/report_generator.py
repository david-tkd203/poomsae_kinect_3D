# src/dataio/report_generator.py
from __future__ import annotations
"""
Generador de reportes de exactitud de calidad de movimiento.

Compara etiquetas GT (CSV) vs predicciones en JSON y exporta un XLSX
con:
- Resumen global
- Métricas por movimiento
- Métricas por video
- Matriz de confusión ponderada por tiempo
- Detalle segmento a segmento
"""

import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

GT_LABELS = ["correcto", "error_leve", "error_grave"]

# ------------------------- util -------------------------

def _video_id_from_path(p: str) -> str:
    try:
        return Path(p).stem
    except Exception:
        return str(p)

def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _extract_segments_from_json(obj: dict) -> Tuple[str, List[dict]]:
    """
    Devuelve (video_id, segmentos) donde segmentos = [{start_s, end_s, label}]

    Soporta esquemas comunes:
      - "segments", "pred_segments", "events", "preds", "windows", "timeline"
      - y ahora también "moves" (como los JSON derivados de CaptureResult)

    También acepta campos:
      - start_s / end_s, t0/t1, start/end, t_start/t_end...
      - label, pred, pred_label, class, prediction, y_hat, quality_label, gt_label...
    """
    # video_id
    vid = None
    for k in ("video_id", "video", "file", "source", "path"):
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            vid = _video_id_from_path(obj[k])
            break

    # claves aceptadas
    start_keys = ("start_s", "t0", "start", "start_sec", "begin", "s", "t_start")
    end_keys   = ("end_s", "t1", "end", "end_sec", "finish", "e", "t_end")
    label_keys = (
        "label",
        "pred",
        "pred_label",
        "class",
        "prediction",
        "y_hat",
        "quality_label",
        "gt_label",
    )

    def grab(d: dict) -> Optional[dict]:
        if not isinstance(d, dict):
            return None
        start = None
        end = None
        lab = None

        for k in start_keys:
            if k in d:
                try:
                    start = float(d[k])
                except Exception:
                    pass
                if start is not None:
                    break

        for k in end_keys:
            if k in d:
                try:
                    end = float(d[k])
                except Exception:
                    pass
                if end is not None:
                    break

        for k in label_keys:
            if k in d:
                lab = str(d[k]).strip().lower()
                break

        if start is not None and end is not None and end > start and lab:
            return {"start_s": float(start), "end_s": float(end), "label": lab}
        return None

    def walk(x) -> List[dict]:
        out: List[dict] = []
        if isinstance(x, dict):
            # intento directo por claves típicas
            for k in (
                "segments",
                "pred_segments",
                "events",
                "preds",
                "windows",
                "timeline",
                "moves",  # soportar JSON de CaptureResult + etiquetas
            ):
                if k in x and isinstance(x[k], list):
                    for it in x[k]:
                        g = grab(it)
                        if g:
                            out.append(g)
            # revisar hijas
            for v in x.values():
                out.extend(walk(v))
        elif isinstance(x, list):
            for it in x:
                if isinstance(it, dict):
                    g = grab(it)
                    if g:
                        out.append(g)
        return out

    segs = walk(obj)

    # normalizar labels a conjunto conocido
    for s in segs:
        if s["label"] not in GT_LABELS:
            if s["label"] in ("leve", "minor", "slight"):
                s["label"] = "error_leve"
            elif s["label"] in ("grave", "major", "severe"):
                s["label"] = "error_grave"
            elif s["label"] in ("ok", "correct", "correcta"):
                s["label"] = "correcto"

    return (vid or "", segs)

def _find_jsons(preds_dir: Path) -> List[Path]:
    return sorted([p for p in preds_dir.rglob("*.json") if p.is_file()])

def _overlap(a0, a1, b0, b1) -> float:
    x0 = max(a0, b0)
    x1 = min(a1, b1)
    return max(0.0, x1 - x0)

# ------------------- carga de datos -------------------

def load_labels(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    # columnas mínimas
    req = {"video", "start_s", "end_s", "label"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"Faltan columnas en labels CSV: {missing}")
    # normalización
    df["video_id"] = df["video"].astype(str).apply(_video_id_from_path)
    df["start_s"] = df["start_s"].astype(float)
    df["end_s"] = df["end_s"].astype(float)
    df["duration"] = df["end_s"] - df["start_s"]
    df = df[df["duration"] > 0]
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    if "move_id" not in df.columns:
        df["move_id"] = ""
    # mapear variantes a conjunto base
    df.loc[df["label"].isin(["leve", "minor", "slight"]), "label"] = "error_leve"
    df.loc[df["label"].isin(["grave", "major", "severe"]), "label"] = "error_grave"
    df.loc[df["label"].isin(["ok", "correct", "correcta"]), "label"] = "correcto"
    return df

def load_predictions(preds_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for jp in _find_jsons(preds_dir):
        obj = _read_json(jp)
        if not obj:
            continue
        vid, segs = _extract_segments_from_json(obj)
        if not segs:
            # fallback: intenta inferir video_id del nombre del json
            if not vid:
                vid = jp.stem
        for s in segs:
            rows.append(
                {
                    "json": str(jp),
                    "video_id": vid or jp.stem,
                    "start_s": float(s["start_s"]),
                    "end_s": float(s["end_s"]),
                    "label": str(s["label"]).lower().strip(),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["json", "video_id", "start_s", "end_s", "label"])
    df = pd.DataFrame(rows)
    df["duration"] = df["end_s"] - df["start_s"]
    df = df[df["duration"] > 0]
    return df

# --------------- emparejamiento y métricas ---------------

def match_and_score(gt_df: pd.DataFrame, pr_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Regresa:
      details_df: una fila por segmento GT con predicción mayoritaria y métricas de solape
      metrics: dict con resúmenes (global, por video, por move_id, confusion)

    Reglas:
      - Para cada segmento GT, la etiqueta predicha es la de mayor tiempo de solape con PR.
      - Exactitud por-segmento = 1 si etiqueta mayoritaria coincide con GT.
      - Exactitud tiempo-ponderada = sum tiempo solape con etiqueta correcta / duración total GT.
    """
    if gt_df.empty:
        raise SystemExit("Labels vacíos.")

    vids = sorted(set(gt_df["video_id"]))
    details: List[dict] = []

    # matriz de confusión tiempo-ponderada
    lab_to_idx = {l: i for i, l in enumerate(GT_LABELS)}
    conf = np.zeros((len(GT_LABELS), len(GT_LABELS)), dtype=float)  # rows = GT, cols = Pred (por tiempo)

    for vid in vids:
        gtv = gt_df[gt_df["video_id"] == vid].copy()
        prv = pr_df[pr_df["video_id"] == vid].copy()

        # si no hay predicciones para este video, todas fallan
        for _, r in gtv.iterrows():
            gt0, gt1, gt_lab, mv = float(r.start_s), float(r.end_s), r.label, r.move_id
            dur = gt1 - gt0
            if prv.empty:
                details.append(
                    {
                        "video_id": vid,
                        "move_id": mv,
                        "gt_start": gt0,
                        "gt_end": gt1,
                        "gt_label": gt_lab,
                        "pred_label": "(sin_pred)",
                        "overlap_total": 0.0,
                        "overlap_correct": 0.0,
                        "overlap_ratio": 0.0,
                        "segment_hit": 0,
                    }
                )
                continue

            # buscar solapes
            prc = prv[(prv["start_s"] < gt1) & (prv["end_s"] > gt0)].copy()
            if prc.empty:
                details.append(
                    {
                        "video_id": vid,
                        "move_id": mv,
                        "gt_start": gt0,
                        "gt_end": gt1,
                        "gt_label": gt_lab,
                        "pred_label": "(sin_pred)",
                        "overlap_total": 0.0,
                        "overlap_correct": 0.0,
                        "overlap_ratio": 0.0,
                        "segment_hit": 0,
                    }
                )
                continue

            # acumular solape por etiqueta
            solape_por_label: Dict[str, float] = {}
            total_ov = 0.0
            correct_ov = 0.0
            for _, p in prc.iterrows():
                ov = _overlap(gt0, gt1, float(p.start_s), float(p.end_s))
                if ov <= 0:
                    continue
                total_ov += ov
                lab = str(p.label)
                solape_por_label[lab] = solape_por_label.get(lab, 0.0) + ov
                if lab == gt_lab:
                    correct_ov += ov

            # pred mayoritario por tiempo
            if solape_por_label:
                pred_lab = max(solape_por_label.items(), key=lambda kv: kv[1])[0]
            else:
                pred_lab = "(sin_pred)"

            # actualizar matriz de confusión por tiempo
            if pred_lab in lab_to_idx and gt_lab in lab_to_idx:
                conf[lab_to_idx[gt_lab], lab_to_idx[pred_lab]] += solape_por_label.get(
                    pred_lab, 0.0
                )

            hit = int(pred_lab == gt_lab and total_ov > 0)
            details.append(
                {
                    "video_id": vid,
                    "move_id": mv,
                    "gt_start": gt0,
                    "gt_end": gt1,
                    "gt_label": gt_lab,
                    "pred_label": pred_lab,
                    "overlap_total": total_ov,
                    "overlap_correct": correct_ov,
                    "overlap_ratio": (correct_ov / dur) if dur > 0 else 0.0,
                    "segment_hit": hit,
                }
            )

    det = pd.DataFrame(details)
    if det.empty:
        raise SystemExit("Sin detalles: revise que existan predicciones JSON compatibles.")

    # métricas globales
    total_seg = len(det)
    seg_acc = float(det["segment_hit"].mean()) if total_seg > 0 else 0.0

    # tiempo-ponderado sobre GT total
    gt_total_time = (gt_df["duration"].sum()) if len(gt_df) else 0.0
    correct_time = float(det["overlap_correct"].sum())
    time_acc = (correct_time / gt_total_time) if gt_total_time > 0 else 0.0

    # por video
    by_video = []
    for vid, g in det.groupby("video_id"):
        idx = gt_df["video_id"] == vid
        gt_time = gt_df.loc[idx, "duration"].sum()
        bt = {
            "video_id": vid,
            "segments": int(len(g)),
            "segment_acc": float(g["segment_hit"].mean()) if len(g) else 0.0,
            "gt_time_s": float(gt_time),
            "correct_time_s": float(g["overlap_correct"].sum()),
            "time_acc": float(g["overlap_correct"].sum() / gt_time) if gt_time > 0 else 0.0,
        }
        by_video.append(bt)
    df_video = pd.DataFrame(by_video).sort_values(
        ["time_acc", "segment_acc"], ascending=[True, True]
    )

    # por move_id
    df_mv = []
    gt_mv = gt_df.copy()
    gt_mv["move_id"] = gt_mv["move_id"].fillna("").replace("", "(sin_move)")
    det_mv = det.copy()
    det_mv["move_id"] = det_mv["move_id"].fillna("").replace("", "(sin_move)")

    for mv, g in det_mv.groupby("move_id"):
        idxm = gt_mv["move_id"] == mv
        gt_time = gt_mv.loc[idxm, "duration"].sum()
        row = {
            "move_id": mv,
            "segments": int(len(g)),
            "segment_acc": float(g["segment_hit"].mean()) if len(g) else 0.0,
            "gt_time_s": float(gt_time),
            "correct_time_s": float(g["overlap_correct"].sum()),
            "time_acc": float(g["overlap_correct"].sum() / gt_time) if gt_time > 0 else 0.0,
        }
        df_mv.append(row)
    df_move = pd.DataFrame(df_mv).sort_values(
        ["time_acc", "segment_acc"], ascending=[True, True]
    )

    # matriz de confusión normalizada por fila (tiempo)
    conf_df = pd.DataFrame(conf, index=GT_LABELS, columns=GT_LABELS)
    conf_df_pct = conf_df.div(conf_df.sum(axis=1).replace(0, np.nan), axis=0)

    metrics = {
        "global": {
            "segments": int(total_seg),
            "segment_acc": float(seg_acc),
            "gt_time_s": float(gt_total_time),
            "correct_time_s": float(correct_time),
            "time_acc": float(time_acc),
        },
        "by_video": df_video,
        "by_move": df_move,
        "confusion_time": conf_df,
        "confusion_time_pct": conf_df_pct,
    }
    return det, metrics

# ------------------- export a Excel -------------------

def export_excel(
    out_xlsx: Path,
    metrics: dict,
    details: pd.DataFrame,
    alias: Optional[str] = None,
) -> Path:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        # Resumen
        g = metrics["global"]
        df_summary = pd.DataFrame(
            [
                {
                    "alias": alias or "",
                    "segmentos_gt": g["segments"],
                    "acierto_segmento_%": round(100.0 * g["segment_acc"], 2),
                    "tiempo_gt_s": round(g["gt_time_s"], 3),
                    "tiempo_correcto_s": round(g["correct_time_s"], 3),
                    "acierto_tiempo_%": round(100.0 * g["time_acc"], 2),
                }
            ]
        )
        df_summary.to_excel(xw, sheet_name="Resumen", index=False)

        # Por movimiento
        df_move = metrics["by_move"].copy()
        if not df_move.empty:
            df_move["acierto_segmento_%"] = (df_move["segment_acc"] * 100).round(2)
            df_move["acierto_tiempo_%"] = (df_move["time_acc"] * 100).round(2)
            cols = [
                "move_id",
                "segments",
                "gt_time_s",
                "correct_time_s",
                "acierto_segmento_%",
                "acierto_tiempo_%",
            ]
            df_move[cols].to_excel(xw, sheet_name="PorMovimiento", index=False)

        # Por video
        df_video = metrics["by_video"].copy()
        if not df_video.empty:
            df_video["acierto_segmento_%"] = (df_video["segment_acc"] * 100).round(2)
            df_video["acierto_tiempo_%"] = (df_video["time_acc"] * 100).round(2)
            cols = [
                "video_id",
                "segments",
                "gt_time_s",
                "correct_time_s",
                "acierto_segmento_%",
                "acierto_tiempo_%",
            ]
            df_video[cols].to_excel(xw, sheet_name="PorVideo", index=False)

        # Confusión por tiempo (valores y %)
        metrics["confusion_time"].to_excel(
            xw, sheet_name="Confusion_tiempo", index=True
        )
        (metrics["confusion_time_pct"] * 100.0).round(2).to_excel(
            xw, sheet_name="Confusion_tiempo_%", index=True
        )

        # Detalles
        det = details.copy()
        det["overlap_ratio_%"] = (det["overlap_ratio"] * 100).round(2)
        det["segment_hit"] = det["segment_hit"].astype(int)
        cols = [
            "video_id",
            "move_id",
            "gt_start",
            "gt_end",
            "gt_label",
            "pred_label",
            "overlap_total",
            "overlap_correct",
            "overlap_ratio_%",
            "segment_hit",
        ]
        det[cols].to_excel(xw, sheet_name="Detalles", index=False)
    return out_xlsx

# ------------------- API de alto nivel -------------------

def generate_report(
    labels_csv: Path | str,
    preds_dir: Path | str,
    out_xlsx: Path | str,
    alias: str = "",
) -> Path:
    """
    Función de alto nivel para usar desde otros módulos del proyecto.

    Args:
        labels_csv: CSV con GT (video,start_s,end_s,label[,move_id])
        preds_dir:  Carpeta con JSON de predicción
        out_xlsx:   Ruta del XLSX de salida
        alias:      Nombre del modelo/configuración (solo informativo)

    Return:
        Ruta al XLSX generado.
    """
    labels_csv = Path(labels_csv)
    preds_dir = Path(preds_dir)
    out_xlsx = Path(out_xlsx)

    gt = load_labels(labels_csv)
    pr = load_predictions(preds_dir)

    if pr.empty:
        raise SystemExit(f"Sin JSON de predicción en: {preds_dir}")

    # filtrar por intersección de videos
    vids_gt = set(gt["video_id"])
    vids_pr = set(pr["video_id"])
    vids_common = vids_gt & vids_pr
    if not vids_common:
        raise SystemExit(
            "No hay intersección de videos entre labels y predicciones. "
            "Verifique video_id/nombres."
        )

    gt = gt[gt["video_id"].isin(vids_common)].reset_index(drop=True)
    pr = pr[pr["video_id"].isin(vids_common)].reset_index(drop=True)

    details, metrics = match_and_score(gt, pr)
    out = export_excel(out_xlsx, metrics, details, alias=alias or None)
    return out

# ------------------- CLI -------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Genera informe de % de acertividad comparando etiquetas GT "
            "vs predicciones JSON."
        )
    )
    ap.add_argument(
        "--labels",
        required=True,
        help="CSV con columnas: video,start_s,end_s,label[,move_id]",
    )
    ap.add_argument(
        "--preds-dir",
        required=True,
        help="Carpeta con JSON de predicción (recursivo)",
    )
    ap.add_argument(
        "--alias",
        type=str,
        default="",
        help="Solo informativo (se incluye en hoja Resumen)",
    )
    ap.add_argument("--out-xlsx", required=True, help="Ruta XLSX de salida")
    args = ap.parse_args()

    labels_csv = Path(args.labels)
    preds_dir = Path(args.preds_dir)
    out_xlsx = Path(args.out_xlsx)

    out = generate_report(labels_csv, preds_dir, out_xlsx, alias=args.alias)
    print(f"[OK] Reporte -> {out}")

if __name__ == "__main__":
    main()
