# src/dataio/score_exporter.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from ..segmentation.move_capture import CaptureResult


# ------------------ Helpers de score ------------------ #

def _score_to_label(score: float) -> str:
    """
    Mapea el match_score [0,1] a una etiqueta cualitativa:

      - >= 0.80  -> "correcto"
      - 0.50–0.80 -> "error_leve"
      - < 0.50   -> "error_grave"
    """
    if score >= 0.80:
        return "correcto"
    if score >= 0.50:
        return "error_leve"
    return "error_grave"


def _label_to_penalty(label: str) -> float:
    """
    Convierte la categoría a una deducción Pal Yang aproximada.
    """
    if label == "error_leve":
        return 0.3
    if label == "error_grave":
        return 0.5
    return 0.0


def _get_move_spec(result: CaptureResult, idx: int) -> Optional[Dict[str, Any]]:
    """
    Busca en result.poomsae_spec (si existe) la spec del movimiento 'idx'.
    Esto nos permite recuperar tech_es, stance_expect, etc.
    """
    spec = getattr(result, "poomsae_spec", None)
    if not isinstance(spec, dict):
        return None

    moves = spec.get("moves") or spec.get("segments") or []
    for mv in moves:
        try:
            if int(mv.get("idx", -1)) == int(idx):
                return mv
        except Exception:
            continue
    return None


# ------------------ Export principal ------------------ #

def export_score_excel_for_sample(
    result: CaptureResult,
    video_id: str,
    out_xlsx: Path,
) -> Path:
    """
    Genera un Excel tipo "<sample_id>_score_final.xlsx" con:

      - Hoja 'detalle': una fila por movimiento.
      - Hoja 'resumen': métricas agregadas tipo Pal Yang.

    La lógica de score por ahora concentra la deducción en las piernas
    (comp_legs), usando el match_score como proxy.
    """
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    cum_penalty = 0.0

    for m in result.moves:
        match_score = float(getattr(m, "match_score", 0.0))
        label = _score_to_label(match_score)
        spec = _get_move_spec(result, getattr(m, "idx", -1))

        tech_kor = spec.get("tech_kor", "") if spec else ""
        tech_es = spec.get("tech_es", "") if spec else getattr(m, "tech_name", "")

        # Código de movimiento ("1a", "1b", etc.) si existe en la spec
        mv_code = None
        if spec:
            mv_code = (
                spec.get("code")
                or spec.get("M")
                or spec.get("id_str")
                or spec.get("id")
            )
        if not mv_code:
            mv_code = str(getattr(m, "idx", ""))

        # En esta versión simple concentramos la deducción en piernas
        comp_arms = "correcto"
        comp_kick = "correcto"
        comp_legs = label  # correcto / error_leve / error_grave

        arms_reason = ""
        kick_reason = ""
        if label == "correcto":
            legs_reason = "match_pose>=0.80"
        elif label == "error_leve":
            legs_reason = "desviacion_leve_en_match_pose"
        else:
            legs_reason = "desviacion_grave_en_match_pose"

        pen_arms = 0.0
        pen_kick = 0.0
        pen_legs = _label_to_penalty(comp_legs)
        ded_total_move = pen_arms + pen_legs + pen_kick
        cum_penalty += ded_total_move
        # Exactitud parte en 4.0 y se descuenta por deducciones acumuladas
        exactitud_acum = max(0.0, 4.0 - cum_penalty)

        stance_exp = spec.get("stance_expect", "") if spec else ""
        turn_exp = spec.get("turn_expect", "") if spec else ""
        kick_exp = spec.get("kick_type", "") if spec else ""

        row = {
            "video_id": video_id,
            "M": mv_code,
            "tech_kor": tech_kor,
            "tech_es": tech_es,
            "comp_arms": comp_arms,
            "comp_legs": comp_legs,
            "comp_kick": comp_kick,
            "arms_reason": arms_reason,
            "legs_reason": legs_reason,
            "kick_reason": kick_reason,
            "pen_arms": pen_arms,
            "pen_legs": pen_legs,
            "pen_kick": pen_kick,
            "ded_total_move": ded_total_move,
            "exactitud_acum": exactitud_acum,
            "stance_exp": stance_exp,
            "stance_pred": getattr(m, "stance_pred", ""),
            "turn_exp": turn_exp,
            "turn_bucket": getattr(m, "rotation_bucket", ""),
            "kick_exp": kick_exp,
            "kick_pred": getattr(m, "kick_pred", ""),
            "match_score": match_score,
            "t0": float(getattr(m, "t_start", 0.0)),
            "t1": float(getattr(m, "t_end", 0.0)),
            "dur_s": float(
                getattr(m, "duration", float(getattr(m, "t_end", 0.0) - getattr(m, "t_start", 0.0)))
            ),
        }
        rows.append(row)

    # ---------- Construir DataFrames detalle y resumen ---------- #
    if not rows:
        # Nada que exportar: devolvemos hojas vacías con columnas estándar
        df_detalle = pd.DataFrame(
            columns=[
                "video_id",
                "M",
                "tech_kor",
                "tech_es",
                "comp_arms",
                "comp_legs",
                "comp_kick",
                "arms_reason",
                "legs_reason",
                "kick_reason",
                "pen_arms",
                "pen_legs",
                "pen_kick",
                "ded_total_move",
                "exactitud_acum",
                "stance_exp",
                "stance_pred",
                "turn_exp",
                "turn_bucket",
                "kick_exp",
                "kick_pred",
                "match_score",
                "t0",
                "t1",
                "dur_s",
            ]
        )
        df_resumen = pd.DataFrame()
    else:
        df_detalle = pd.DataFrame(rows)

        # Número de movimientos esperados según spec (si existe)
        moves_expected = 0
        spec = getattr(result, "poomsae_spec", None)
        if isinstance(spec, dict):
            mv_list = spec.get("moves") or spec.get("segments") or []
            moves_expected = len(mv_list)
        if moves_expected <= 0:
            moves_expected = len(result.moves)

        moves_detected = len(result.moves)
        moves_scored = len(result.moves)
        moves_correct_90p = sum(
            1 for m in result.moves if float(getattr(m, "match_score", 0.0)) >= 0.90
        )

        ded_total = float(df_detalle["ded_total_move"].sum())
        exactitud_final = float(df_detalle["exactitud_acum"].iloc[-1])

        def _pct_ok(col_name: str) -> float:
            if moves_scored == 0:
                return 0.0
            ok = (df_detalle[col_name] == "correcto").sum()
            return round(100.0 * ok / float(moves_scored), 2)

        pct_arms_ok = _pct_ok("comp_arms")
        pct_legs_ok = _pct_ok("comp_legs")
        pct_kick_ok = _pct_ok("comp_kick")

        df_resumen = pd.DataFrame(
            [
                {
                    "video_id": video_id,
                    "moves_expected": moves_expected,
                    "moves_detected": moves_detected,
                    "moves_scored": moves_scored,
                    "moves_correct_90p": moves_correct_90p,
                    "ded_total": ded_total,
                    "exactitud_final": exactitud_final,
                    "pct_arms_ok": pct_arms_ok,
                    "pct_legs_ok": pct_legs_ok,
                    "pct_kick_ok": pct_kick_ok,
                    "restart_penalty": 0.0,
                }
            ]
        )

    # ---------- Escritura a Excel ---------- #
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_detalle.to_excel(writer, sheet_name="detalle", index=False)
        df_resumen.to_excel(writer, sheet_name="resumen", index=False)

    return out_xlsx
