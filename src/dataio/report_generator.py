# src/dataio/report_generator.py
"""Generador de informes PDF a partir de resultados de scoring.

Contiene utilidades para leer los Excel de scoring y componer un
PDF amigable para informes de práctica. Se han añadido pequeñas
mejoras de estilo y corrección de textos para que el informe sea
más legible.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import json

import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ------------------------------------------------------------
# Helpers de carga de datos
# ------------------------------------------------------------

def load_scoring_from_xlsx(score_xlsx: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga las hojas 'detalle' y 'resumen' producidas por score_pal_yang.py.
    Si alguna falta, devuelve DataFrames vacíos con columnas mínimas.
    """
    score_xlsx = Path(score_xlsx)
    if not score_xlsx.exists():
        raise FileNotFoundError(f"No se encontró el archivo de scoring: {score_xlsx}")

    xls = pd.ExcelFile(score_xlsx)

    if "detalle" in xls.sheet_names:
        df_det = pd.read_excel(xls, "detalle")
    else:
        df_det = pd.DataFrame(columns=[
            "video_id", "M", "tech_kor", "tech_es",
            "comp_arms", "comp_legs", "comp_kick",
            "ded_total_move", "exactitud_acum", "move_acc",
            "fail_parts",
        ])

    if "resumen" in xls.sheet_names:
        df_sum = pd.read_excel(xls, "resumen")
    else:
        df_sum = pd.DataFrame(columns=[
            "video_id", "moves_expected", "moves_detected",
            "moves_scored", "moves_correct_90p",
            "ded_total", "exactitud_final",
            "pct_arms_ok", "pct_legs_ok", "pct_kick_ok",
            "restart_penalty",
        ])

    return df_det, df_sum


def load_session_metadata(capture_dir: Path) -> Dict[str, Any]:
    """
    Intenta leer algún JSON de metadatos en la carpeta de captura.
    Nombres posibles: session_meta.json, metadata.json, meta.json
    (opcional; si no hay nada, devuelve {}).
    """
    capture_dir = Path(capture_dir)
    for name in ("session_meta.json", "metadata.json", "meta.json"):
        p = capture_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


# ------------------------------------------------------------
# Lógica de interpretación de resultados
# ------------------------------------------------------------

def describe_score(exactitud_final: float) -> str:
    """
    Traduce la nota (1.5–4.0) a una etiqueta cualitativa.
    Ajusta los cortes según lo que tú quieras en tu tesis.
    """
    if exactitud_final >= 3.8:
        return "Desempeño sobresaliente. La ejecución global del poomsae es altamente consistente."
    if exactitud_final >= 3.5:
        return "Muy buen desempeño. Se observan detalles menores, pero la técnica general es sólida."
    if exactitud_final >= 3.0:
        return "Desempeño adecuado. Existen aspectos técnicos mejorables, aunque la base está bien lograda."
    return "Área de mejora. Se identifican varias desviaciones técnicas que requieren trabajo específico."


def _fmt_pct(v: Any) -> str:
    try:
        if pd.isna(v):
            return "—"
        return f"{float(v):.1f} %"
    except Exception:
        return "—"


def _fmt_num(v: Any, ndigits: int = 2) -> str:
    try:
        if pd.isna(v):
            return "—"
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return "—"


# ------------------------------------------------------------
# Generación de PDF
# ------------------------------------------------------------

def generate_pal_yang_report(
    capture_dir: Path,
    score_xlsx: Path,
    out_pdf: Optional[Path] = None,
) -> Path:
    """
    Genera un informe en PDF a partir del Excel generado por score_pal_yang.py.

    Parámetros
    ----------
    capture_dir : Path
        Carpeta raíz de la sesión (captures/session_YYYYMMDD_HHMMSS).
    score_xlsx : Path
        Ruta al Excel de scoring (con hojas 'detalle' y 'resumen').
    out_pdf : Path, opcional
        Ruta de salida del PDF. Si no se indica, se usa
        <capture_dir>/pal_yang_report.pdf

    Retorna
    -------
    Path
        Ruta final del PDF generado.
    """
    capture_dir = Path(capture_dir)
    score_xlsx = Path(score_xlsx)

    if out_pdf is None:
        out_pdf = capture_dir / "pal_yang_report.pdf"
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    df_det, df_sum = load_scoring_from_xlsx(score_xlsx)
    meta = load_session_metadata(capture_dir)

    #los DataFrames `df_det` y `df_sum` provienen del
    # script de scoring y pueden no contener todas las columnas. Las
    # funciones helper `_fmt_pct` y `_fmt_num` se encargan de formatear
    # valores faltantes de forma elegante.

    # ---------- Estilos ----------
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
    ))
    styles.add(ParagraphStyle(
        name="BodyJustify",
        parent=styles["BodyText"],
        alignment=4,  # TA_JUSTIFY
        leading=14,
    ))

    title_style = styles["Title"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]
    body_small = styles["Small"]
    body_just = styles["BodyJustify"]

    # ---------- Story ----------
    story = []

    # 1) Carátula
    story.append(Paragraph("Informe de evaluación técnica Pal Jang", title_style))
    story.append(Spacer(1, 0.5 * cm))

    today = datetime.now().strftime("%d-%m-%Y %H:%M")
    story.append(Paragraph(f"Fecha de generación: {today}", body))
    story.append(Spacer(1, 0.3 * cm))

    # Metadata básica, si existe
    atleta = meta.get("athlete_name") or meta.get("nombre_atleta") or "No especificado"
    categoria = meta.get("category") or meta.get("categoria") or "No especificada"
    cinturon = meta.get("belt") or meta.get("cinturon") or "No especificado"
    video_id = None

    if not df_sum.empty and "video_id" in df_sum.columns:
        try:
            video_id = str(df_sum["video_id"].iloc[0])
        except Exception:
            pass
    if not video_id and not df_det.empty and "video_id" in df_det.columns:
        try:
            video_id = str(df_det["video_id"].iloc[0])
        except Exception:
            pass

    story.append(Paragraph(f"Atleta: <b>{atleta}</b>", body))
    story.append(Paragraph(f"Categoría: <b>{categoria}</b>", body))
    # Corregimos un typo presente en la versión anterior: 'Citinturón' -> 'Cinturón'
    story.append(Paragraph(f"Cinturón: <b>{cinturon}</b>", body))
    if video_id:
        story.append(Paragraph(f"Video / ID evaluación: <b>{video_id}</b>", body))
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph(
        "Este informe resume los resultados de la evaluación automática del poomsae Pal Jang, "
        "utilizando un modelo de análisis de pose y reglas técnicas derivadas del reglamento "
        "World Taekwondo para Poomsae. La nota se construye a partir de penalizaciones leves "
        "y graves por desviaciones técnicas en brazos, piernas y patadas.",
        body_just,
    ))
    story.append(Spacer(1, 0.6 * cm))

    # 2) Resumen numérico global
    story.append(Paragraph("1. Resumen global de la evaluación", h2))
    story.append(Spacer(1, 0.3 * cm))

    if df_sum.empty:
        story.append(Paragraph(
            "No se encontraron datos de resumen en el archivo de scoring. "
            "Verifique que score_pal_yang.py se haya ejecutado correctamente.",
            body,
        ))
    else:
        s = df_sum.iloc[0]

        exactitud_final = float(s.get("exactitud_final", 0.0) or 0.0)
        ded_total = float(s.get("ded_total", 0.0) or 0.0)
        moves_expected = int(s.get("moves_expected", 0) or 0)
        moves_detected = int(s.get("moves_detected", 0) or 0)
        moves_scored = int(s.get("moves_scored", 0) or 0)
        moves_correct_90p = int(s.get("moves_correct_90p", 0) or 0)
        pct_arms_ok = _fmt_pct(s.get("pct_arms_ok", None))
        pct_legs_ok = _fmt_pct(s.get("pct_legs_ok", None))
        pct_kick_ok = _fmt_pct(s.get("pct_kick_ok", None))
        restart_penalty = float(s.get("restart_penalty", 0.0) or 0.0)

        table_data = [
            ["Indicador", "Valor"],
            ["Nota final (exactitud)", f"{exactitud_final:.2f} / 4.00"],
            ["Deducción total acumulada", _fmt_num(ded_total, 3)],
            ["Penalización por reinicios", _fmt_num(restart_penalty, 2)],
            ["Movimientos esperados", str(moves_expected)],
            ["Movimientos detectados", str(moves_detected)],
            ["Movimientos evaluados", str(moves_scored)],
            ["Movimientos correctos (≥90 %)", str(moves_correct_90p)],
            ["Técnicas de brazos OK", pct_arms_ok],
            ["Posturas de piernas OK", pct_legs_ok],
            ["Patadas OK", pct_kick_ok],
        ]

        tbl = Table(table_data, colWidths=[7 * cm, 7 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.whitesmoke, colors.HexColor("#f0f0f0")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph(describe_score(exactitud_final), body_just))
        story.append(Spacer(1, 0.6 * cm))

    # 3) Detalle por movimiento
    story.append(Paragraph("2. Detalle por movimiento", h2))
    story.append(Spacer(1, 0.3 * cm))

    if df_det.empty:
        story.append(Paragraph(
            "No se encontraron datos de detalle en la hoja 'detalle'.",
            body,
        ))
    else:
        story.append(Paragraph(
            "La siguiente tabla resume, para cada movimiento del poomsae, "
            "el estado de brazos, piernas y patada (cuando aplica), junto con "
            "las partes donde se detectaron desviaciones técnicas.",
            body_just,
        ))
        story.append(Spacer(1, 0.3 * cm))

        # Seleccionamos columnas “clave” para que la tabla no sea gigantesca
        cols_existentes = df_det.columns.tolist()
        cols_preferidas = [
            "M",
            "tech_es",
            "comp_arms",
            "comp_legs",
            "comp_kick",
            "fail_parts",
            "exactitud_acum",
        ]
        cols_usar = [c for c in cols_preferidas if c in cols_existentes]

        # Si no está "tech_es", intentar usar "tech_kor"
        if "tech_es" not in cols_usar and "tech_kor" in cols_existentes:
            cols_usar.insert(1, "tech_kor")

        if not cols_usar:
            story.append(Paragraph(
                "No se encontraron columnas estándar para generar la tabla de detalle.",
                body,
            ))
        else:
            # Construimos la tabla en bloques para no reventar la página
            header = [c.replace("_", " ").capitalize() for c in cols_usar]

            # ordenamos por M si existe
            df_det_sorted = df_det.copy()
            if "M" in df_det_sorted.columns:
                # M suele ser algo tipo '1', '2a', etc.
                df_det_sorted["__order"] = df_det_sorted["M"].astype(str)
                df_det_sorted = df_det_sorted.sort_values("__order")
                df_det_sorted = df_det_sorted.drop(columns=["__order"])

            data_rows = []
            for _, row in df_det_sorted.iterrows():
                r = []
                for col in cols_usar:
                    val = row.get(col, "")
                    if col == "exactitud_acum":
                        r.append(_fmt_num(val, 3))
                    else:
                        r.append(str(val) if val == val else "—")
                data_rows.append(r)

            # Particionamos en bloques de, por ejemplo, 25 movimientos
            chunk_size = 25
            for offset in range(0, len(data_rows), chunk_size):
                chunk = data_rows[offset:offset + chunk_size]
                table_data = [header] + chunk

                tbl = Table(table_data, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.whitesmoke, colors.HexColor("#f5f5f5")]),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]))

                story.append(tbl)
                story.append(Spacer(1, 0.4 * cm))

                # Si todavía quedan filas, meter salto de página
                if offset + chunk_size < len(data_rows):
                    story.append(PageBreak())

    # 4) Notas finales / observaciones
    story.append(PageBreak())
    story.append(Paragraph("3. Observaciones generales", h2))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(
        "Las métricas anteriores deben interpretarse siempre en conjunto con la evaluación "
        "experta del entrenador o juez. El sistema de análisis de pose está sujeto a "
        "limitaciones propias de la captura (posición de la cámara, oclusiones, ropa, "
        "calidad de iluminación) y de la segmentación automática del poomsae.",
        body_just,
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(
        "El objetivo de este informe es proporcionar una referencia cuantitativa para apoyar "
        "la práctica y el entrenamiento, destacando tendencias y patrones de error más que "
        "reemplazar la evaluación humana.",
        body_just,
    ))

    # ---------- Construcción del PDF ----------
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    doc.build(story)

    return out_pdf
