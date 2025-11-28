#!/usr/bin/env python3
"""
unified_analysis.py
--------------------
Script unificado que integra utilidades de análisis de scoring / spec 8yang.

Subcomandos disponibles:
 - summarize     : Resumen rápido del XLSX de scoring
 - scoring       : Resumen y estadísticas de scoring
 - fixed         : Análisis de scoring con umbrales corregidos
 - arms_cause    : Análisis de causas de 'GRAVE' en brazos
 - level_grave   : Análisis de niveles en movimientos marcados como GRAVE
 - missing_moves : Análisis de movimientos faltantes (filas 28-37 de la spec)
 - rotation      : Comparación giro esperado vs medido
 - check_spec    : Inspeccionar la especificación JSON
 - check_moves   : Contar movimientos en archivos JSON de anotaciones
 - check_turns   : Resumen de valores 'turn' en la especificación
 - debug_level   : Debug detallado de medidas de nivel (matrices/estadísticas)
 - debug_rel     : Debug del cálculo de niveles relativos desde CSV de landmarks
 - all           : Ejecuta un set razonable de análisis (scoring, level, rot, arms)

Pensado para ejecutarse desde la raíz del repo:
  python -m src.tools.unified_analysis summarize --xlsx reports/8yang_001_CORRECTED.xlsx
  python -m src.tools.unified_analysis rotation  --xlsx reports/8yang_001_CORRECTED.xlsx --spec config/patterns/8yang_spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, List

import pandas as pd

# ---------------------------------------------------------------------
# Localización del proyecto
#   Este archivo vive en:  src/tools/unified_analysis.py
#   parents[0] = .../src/tools
#   parents[1] = .../src
#   parents[2] = .../poomsae_kinect_3d   <-- raíz del proyecto
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]

# Aseguramos que Python ve tanto la raíz (para `import src.*`)
# como src directamente (para `import segmentation`, etc.)
for p in (PROJECT_ROOT, SRC_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def summarize_xlsx(xlsx_path: Path) -> None:
    """Resumen rápido del XLSX de scoring."""
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return

    xl = pd.ExcelFile(xlsx_path)
    print(f"Hojas disponibles: {xl.sheet_names}\n")

    df = pd.read_excel(xlsx_path, sheet_name=0)
    print("=" * 80)
    print(f"RESUMEN: {xlsx_path.name}")
    print("=" * 80)
    print(f"Total filas: {len(df)}")
    print(f"Columnas: {list(df.columns)}\n")

    # Intentar detectar columna de exactitud
    acc_col = None
    for c in df.columns:
        if "exactitud" in c.lower() or "acc" in c.lower():
            acc_col = c
            break
    if acc_col:
        vals = pd.to_numeric(df[acc_col], errors="coerce")
        print(f"Exactitud promedio ({acc_col}): {vals.mean():.4f}")
        print(f"Exactitud min/max: {vals.min():.4f}/{vals.max():.4f}\n")
    else:
        print("Columna de exactitud no encontrada\n")

    print(df.head(10).to_string())


def analyze_scoring(xlsx_path: Path) -> None:
    """Analiza y resume indicadores del scoring."""
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return
    df = pd.read_excel(xlsx_path)
    print("=" * 80)
    print("RESUMEN DE SCORING")
    print("=" * 80)
    print(f"Total movimientos: {len(df)}")
    yes = (df["is_correct"] == "yes").sum() if "is_correct" in df.columns else 0
    print(f"Correctos (is_correct=yes): {yes}")

    # Componentes
    for col in ["comp_arms", "comp_legs", "comp_kick"]:
        if col in df.columns:
            print(f"\n{col}:")
            counts = df[col].value_counts()
            for val, cnt in counts.items():
                pct = cnt / len(df) * 100
                print(f"  {val:10s}: {cnt:3d} ({pct:5.1f}%)")

    # Deducciones
    for k in ["pen_arms", "pen_legs", "pen_kick", "ded_total_move"]:
        if k in df.columns:
            print(f"\n{k}: total={df[k].sum():.2f}, mean={df[k].mean():.4f}")


def analyze_fixed_scoring(xlsx_path: Path) -> None:
    """Análisis específico para archivos con umbrales corregidos."""
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return
    df = pd.read_excel(xlsx_path)
    print("=" * 80)
    print("ANÁLISIS (UMBR. CORREGIDOS)")
    print("=" * 80)
    print(f"Total movimientos: {len(df)}")
    col = "is_correct"
    print(
        f"Movimientos correctos: "
        f"{(df[col] == 'yes').sum() if col in df.columns else 'N/A'}"
    )
    if "level(exp)" in df.columns and "level(meas)" in df.columns:
        print("\nMatriz de confusión (level(exp) vs level(meas))")
        print(pd.crosstab(df["level(exp)"], df["level(meas)"], margins=True))


def analyze_arm_grave_cause(xlsx_path: Path) -> None:
    """Analizar por qué hay GRAVE en brazos."""
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return
    df = pd.read_excel(xlsx_path)
    df_grave = df[df.get("comp_arms", pd.Series(dtype=object)) == "GRAVE"]
    print("=" * 80)
    print("CAUSA RAÍZ: GRAVE EN BRAZOS")
    print("=" * 80)
    cols = [
        "M",
        "tech_es",
        "level(meas)",
        "rot(meas_deg)",
        "elbow_median_deg",
        "fail_parts",
    ]
    for c in cols:
        if c not in df.columns:
            print(f"Aviso: columna {c} no encontrada en el XLSX")
    if len(df_grave) == 0:
        print("No hay filas con comp_arms == 'GRAVE'")
        return
    print(df_grave[[c for c in cols if c in df.columns]].to_string())


def analyze_level_grave(xlsx_path: Path) -> None:
    """Analizar qué niveles se miden cuando hay GRAVE en brazos."""
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return
    df = pd.read_excel(xlsx_path)
    print("=" * 80)
    print("ANÁLISIS DE NIVELES EN MOVIMIENTOS GRAVE")
    print("=" * 80)
    df_grave = df[df.get("comp_arms", pd.Series(dtype=object)) == "GRAVE"]
    print(f"Total GRAVE: {len(df_grave)}")

    if "level(exp)" in df.columns:
        print("\nDistribución level(exp):")
        print(df_grave["level(exp)"].value_counts())
    if "level(meas)" in df.columns:
        print("\nDistribución level(meas):")
        print(df_grave["level(meas)"].value_counts())
    if "y_rel_end" in df.columns:
        yv = df_grave["y_rel_end"].dropna()
        if len(yv) > 0:
            print("\nEstadísticas y_rel_end:")
            print(yv.describe())


def analyze_missing_moves(spec_path: Path, xlsx_path: Path, seg_report: Optional[Path] = None) -> None:
    """Analiza movimientos esperados (filas 28-37) y compara con segmentación."""
    if not spec_path.exists():
        print(f"❌ spec no encontrado: {spec_path}")
        return
    if not xlsx_path.exists():
        print(f"❌ xlsx no encontrado: {xlsx_path}")
        return

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    moves = spec.get("moves", [])
    print("=" * 80)
    print("MOVIMIENTOS ESPERADOS (filas 28-37)")
    print("=" * 80)
    for row_idx in range(27, 37):
        if row_idx < len(moves):
            m = moves[row_idx]
            print(
                f"Fila {row_idx+1}: M={m.get('seq')} "
                f"({m.get('idx')}{m.get('sub','')}) | "
                f"Extremidad: {m.get('active_limb')} | "
                f"Patada: {m.get('kick_type','none')}"
            )

    if seg_report and seg_report.exists():
        seg = json.loads(seg_report.read_text(encoding="utf-8"))
        print("\nSegmentación report:")
        print(f"Total segmentos detectados: {seg.get('total_segments')}")
        if "segments" in seg:
            print("Últimos segmentos detectados:")
            for s in seg["segments"][-7:]:
                print(
                    f"  #{s.get('id')}: t={s.get('time')} "
                    f"extremidad={s.get('active_limb')} kick={s.get('kick')}"
                )


def analyze_rotation(xlsx_path: Path, spec_path: Path) -> None:
    """Comparar rotaciones esperadas en spec y medidas en XLSX."""
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return
    if not spec_path.exists():
        print(f"❌ spec no encontrado: {spec_path}")
        return

    df = pd.read_excel(xlsx_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_map = {
        f"{m.get('idx')}{m.get('sub','')}": m for m in spec.get("moves", [])
    }
    turns_deg = {
        "NONE": 0,
        "LEFT_90": -90,
        "RIGHT_90": 90,
        "LEFT_180": 180,
        "RIGHT_180": 180,
        "LEFT_270": -270,
        "RIGHT_270": 270,
    }

    print("=" * 80)
    print("ANÁLISIS ROTACIONES")
    print("=" * 80)
    print("Move | exp_turn | exp_deg | meas_deg | diff_deg | comp_arms")

    for _, row in df.iterrows():
        m_key = row.get("M")
        if not m_key or m_key not in spec_map:
            continue
        spec_m = spec_map[m_key]
        turn_code = spec_m.get("turn", "NONE")
        exp_deg = turns_deg.get(turn_code, 0)
        meas_deg = row.get("rot(meas_deg)", 0)

        try:
            diff = abs(meas_deg - exp_deg)
            if diff > 180:
                diff = 360 - diff
        except Exception:
            diff = float("nan")

        print(
            f"{m_key:4s} | {turn_code:10s} | {exp_deg:7.1f} | "
            f"{meas_deg:8.2f} | {diff:7.1f} | {row.get('comp_arms','-')}"
        )


def check_spec(spec_path: Path) -> None:
    if not spec_path.exists():
        print(f"❌ No encontrado: {spec_path}")
        return
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    print(f"Clave 'moves': {len(spec.get('moves', []))} movimientos")
    if spec.get("moves"):
        first = spec["moves"][0]
        print(f"Keys primer movimiento: {list(first.keys())}")


def check_moves(files: List[Path]) -> None:
    for f in files:
        p = Path(f)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                print(f"{p}: {len(data.get('moves', []))} movimientos")
            except Exception as e:
                print(f"Error leyendo {p}: {e}")
        else:
            print(f"{p}: NO EXISTS")


def check_turns(spec_path: Path) -> None:
    if not spec_path.exists():
        print(f"❌ No encontrado: {spec_path}")
        return
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    turns: dict[str, int] = {}
    for m in spec.get("moves", []):
        t = m.get("turn", "NONE")
        turns[t] = turns.get(t, 0) + 1
    print("Valores de 'turn' en spec:")
    for t, c in sorted(turns.items()):
        print(f"  {t:20s}: {c}")


def debug_level_meas(xlsx_path: Path) -> None:
    """Debug de niveles y matriz de confusión, estadísticas por level(exp)."""
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return
    df = pd.read_excel(xlsx_path)
    print("=" * 80)
    print("DEBUG LEVEL MEAS")
    print("=" * 80)
    if "level(exp)" in df.columns and "level(meas)" in df.columns:
        print(pd.crosstab(df["level(exp)"], df["level(meas)"], margins=True))
    for level in ["ARAE", "MOMTONG", "OLGUL"]:
        subset = (
            df[df.get("level(exp)") == level]
            if "level(exp)" in df.columns
            else pd.DataFrame()
        )
        if len(subset) > 0 and "y_rel_end" in subset.columns:
            y = subset["y_rel_end"].dropna()
            print(
                f"\n{level}: n={len(subset)}, measured={len(y)} -> "
                f"min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}"
            )


def debug_rel_calc(csv_path: Path) -> None:
    """Verificar cálculo de niveles relativos a partir de CSV de landmarks.

    Intenta usar primero el loader robusto de `src.tools.analyze_poomsae_metrics`,
    y si no está disponible, cae a `segmentation.move_capture.load_landmarks_csv`
    del segmentador original.
    """
    if not csv_path.exists():
        print(f"❌ No encontrado: {csv_path}")
        return

    load_fn = None

    # 1) Nuevo loader (t, xyz) del análisis de energía
    try:
        from src.tools.analyze_poomsae_metrics import load_landmarks_csv as _l

        load_fn = _l
        loader_type = "metrics"
    except Exception:
        load_fn = None
        loader_type = ""

    # 2) Loader del segmentador original (DataFrame)
    if load_fn is None:
        try:
            from segmentation.move_capture import load_landmarks_csv as _l  # type: ignore

            load_fn = _l
            loader_type = "segmentation"
        except Exception as e:
            print(
                "No se pudo importar ningún load_landmarks_csv "
                "(ni src.tools.analyze_poomsae_metrics ni segmentation.move_capture):",
                e,
            )
            return

    print(f"[debug_rel] usando loader tipo: {loader_type}")
    res = load_fn(csv_path)

    # Si el loader devuelve (t, xyz)
    if isinstance(res, tuple) and len(res) == 2:
        t, xyz = res
        print(f"t shape: {t.shape}, xyz shape: {xyz.shape}")
        print("Primeros 5 tiempos (s):", t[:5])
        if xyz.ndim == 3:
            print(f"N frames: {xyz.shape[0]}, N joints: {xyz.shape[1]}")
    else:
        # Asumimos que es un DataFrame
        try:
            df = res  # type: ignore
            print("Estructura de landmarks (head):")
            print(df.head().to_string())
        except Exception:
            print("Resultado de load_landmarks_csv no reconocido:", type(res))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unified_analysis", description="Herramienta unificada de análisis 8yang"
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("summarize", help="Resumen rápido del XLSX")
    sub.add_parser("scoring", help="Resumen de scoring")
    sub.add_parser("fixed", help="Análisis de scoring con umbrales corregidos")
    sub.add_parser("arms_cause", help="Analizar causa de GRAVE en brazos")
    sub.add_parser("level_grave", help="Analizar niveles en movimientos GRAVE")
    sub.add_parser("missing_moves", help="Analizar movimientos faltantes (filas 28-37)")
    sub.add_parser("rotation", help="Analizar rotaciones spec vs meas")
    sub.add_parser("check_spec", help="Inspeccionar la spec JSON")
    sub.add_parser("check_moves", help="Contar movimientos en archivos JSON")
    sub.add_parser("check_turns", help="Listar valores de turn en spec")
    sub.add_parser("debug_level", help="Debug de level measurements")
    sub.add_parser("debug_rel", help="Debug del cálculo relativo a partir de CSV")
    sub.add_parser(
        "all", help="Ejecutar varios análisis (scoring, level, rotation, arms)"
    )

    p.add_argument(
        "--xlsx",
        type=Path,
        help=(
            "Archivo XLSX de scoring "
            "(por defecto reports/8yang_001_CORRECTED.xlsx)"
        ),
    )
    p.add_argument(
        "--spec",
        type=Path,
        help="Spec JSON (por defecto config/patterns/8yang_spec.json)",
    )
    p.add_argument(
        "--seg-report",
        type=Path,
        help="Reporte de segmentación JSON (opcional, para missing_moves)",
    )
    p.add_argument(
        "--moves-files",
        type=Path,
        nargs="*",
        help="Lista de archivos moves JSON para check_moves",
    )
    p.add_argument(
        "--csv",
        type=Path,
        help="CSV de landmarks para debug_rel",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    # Paths por defecto
    xlsx = args.xlsx or (PROJECT_ROOT / "reports" / "8yang_001_CORRECTED.xlsx")
    spec = args.spec or (PROJECT_ROOT / "config" / "patterns" / "8yang_spec.json")

    if args.cmd == "summarize":
        summarize_xlsx(Path(xlsx))
    elif args.cmd == "scoring":
        analyze_scoring(Path(xlsx))
    elif args.cmd == "fixed":
        analyze_fixed_scoring(Path(xlsx))
    elif args.cmd == "arms_cause":
        analyze_arm_grave_cause(Path(xlsx))
    elif args.cmd == "level_grave":
        analyze_level_grave(Path(xlsx))
    elif args.cmd == "missing_moves":
        analyze_missing_moves(Path(spec), Path(xlsx), args.seg_report)
    elif args.cmd == "rotation":
        analyze_rotation(Path(xlsx), Path(spec))
    elif args.cmd == "check_spec":
        check_spec(Path(spec))
    elif args.cmd == "check_moves":
        default_moves = PROJECT_ROOT / "data" / "moves_ml" / "8yang_001_moves.json"
        files = args.moves_files or [default_moves]
        check_moves(files)
    elif args.cmd == "check_turns":
        check_turns(Path(spec))
    elif args.cmd == "debug_level":
        debug_level_meas(Path(xlsx))
    elif args.cmd == "debug_rel":
        if args.csv is None:
            print("Especifica --csv path/to/landmarks.csv para debug_rel")
        else:
            debug_rel_calc(Path(args.csv))
    elif args.cmd == "all":
        analyze_scoring(Path(xlsx))
        analyze_level_grave(Path(xlsx))
        analyze_rotation(Path(xlsx), Path(spec))
        analyze_arm_grave_cause(Path(xlsx))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
