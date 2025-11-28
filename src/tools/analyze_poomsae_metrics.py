# src/tools/analyze_poomsae_metrics.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy es opcional; si no existe, simplemente no marcamos picos
try:
    from scipy.signal import find_peaks
except Exception:  # ImportError u otros
    find_peaks = None  # type: ignore


# ======================================================================
# CONFIG GENERAL
# ======================================================================

# Se asume que los CSV de landmarks están a ~60 FPS
FPS_HINT = 60.0

ROOT = Path(__file__).resolve().parents[2]


# ======================================================================
# CARGA ROBUSTA DE LANDMARKS (CSV)
# ======================================================================

TIME_COL_CANDIDATES = ["t", "time", "timestamp", "time_s", "time_sec"]
FRAME_COL_CANDIDATES = ["frame", "frame_idx", "frame_index"]
META_COLS = {
    "sample_id",
    "video_id",
    "clip",
    "label",
    "side",
    "poomsae",
    "subset",
}


def _detect_time_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Devuelve:
      t  : np.ndarray (tiempos en segundos desde 0)
      df : dataframe sin la columna de tiempo/frames

    Heurística:
      - Si hay columna de tiempo (t / time / ...):
          * si parece frames enteros con paso ~1 -> frames/FPS_HINT
          * si parece segundos -> se usa tal cual (normalizado a t[0]=0)
      - Si hay columna de frame -> frames/FPS_HINT
      - Si no hay nada -> índice/FPS_HINT
    """
    # 1) columnas explícitas de tiempo
    for col in TIME_COL_CANDIDATES:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() < len(df) * 0.5:
                continue
            raw = s.to_numpy(dtype=np.float64)

            diffs = np.diff(raw)
            # ¿parece frames? (enteros, paso ~1)
            if np.allclose(raw, np.round(raw), atol=1e-3) and np.allclose(
                diffs, np.ones_like(diffs), atol=1e-2
            ):
                frames = raw - raw[0]
                t = frames / max(FPS_HINT, 1.0)
                print(f"[time] '{col}' detectado como FRAMES (fps≈{FPS_HINT})")
            else:
                t = raw - raw[0]
                print(f"[time] '{col}' detectado como TIEMPO en segundos")

            df = df.drop(columns=[col])
            return t.astype(np.float32), df

    # 2) columnas de frame
    for col in FRAME_COL_CANDIDATES:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() < len(df) * 0.5:
                continue
            frames = s.to_numpy(dtype=np.float64)
            frames = frames - frames[0]
            t = frames / max(FPS_HINT, 1.0)
            df = df.drop(columns=[col])
            print(f"[time] '{col}' detectado como FRAMES (fps≈{FPS_HINT})")
            return t.astype(np.float32), df

    # 3) fallback: índice como frames
    idx = np.arange(len(df), dtype=np.float64)
    t = idx / max(FPS_HINT, 1.0)
    print(f"[time] sin columna explícita → usando índice/FPS_HINT (fps≈{FPS_HINT})")
    return t.astype(np.float32), df


def load_landmarks_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga un CSV de landmarks y devuelve:
      t   : (N,)  en segundos
      xyz : (N, J, 3) posiciones 3D

    Robusto a:
      - columnas de texto (sample_id, etc.)
      - columnas numéricas constantes (fps, etc.)
      - nº de columnas no múltiplo de 3 (se recortan por la derecha).
    """
    df = pd.read_csv(csv_path)

    # Quitar columnas meta conocidas
    df = df.drop(columns=[c for c in META_COLS if c in df.columns], errors="ignore")

    # Detectar tiempo / frames
    t, df = _detect_time_from_df(df)

    # Solo columnas numéricas
    num_df = df.select_dtypes(include=[np.number]).copy()

    # Quitar columnas completamente NaN
    num_df = num_df.dropna(axis=1, how="all")

    # Quitar columnas constantes (ej. fps)
    nunique = num_df.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"[cols] Eliminando columnas constantes en {csv_path.name}: {const_cols}")
        num_df = num_df.drop(columns=const_cols)

    if num_df.shape[1] < 3:
        raise ValueError(
            f"El CSV {csv_path} tiene solo {num_df.shape[1]} columnas numéricas "
            f"tras limpiar; se necesitan ≥3 para (x,y,z)."
        )

    n_cols = num_df.shape[1]
    if n_cols % 3 != 0:
        n_triplets = n_cols // 3
        used = n_triplets * 3
        print(
            f"[WARN] {csv_path} tiene {n_cols} columnas numéricas "
            f"(no múltiplo de 3). Usaré solo las primeras {used} "
            f"para formar {n_triplets} joints (x,y,z) y descarto {n_cols - used}."
        )
        num_df = num_df.iloc[:, :used]
        n_cols = used

    n_joints = n_cols // 3
    feat = num_df.to_numpy(dtype=np.float32)
    xyz = feat.reshape(len(num_df), n_joints, 3)

    return t, xyz


# ======================================================================
# ENERGÍA CINÉTICA Y COMPONENTES DE ENERGÍA
# ======================================================================

def compute_kinetic_energy(t: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Energía cinética global (proxy) a partir de velocidades de todos
    los joints. No usa masas reales: solo sirve como métrica relativa.
    """
    vel = np.gradient(xyz, t, axis=0)          # dxyz/dt
    speed2 = np.sum(vel ** 2, axis=2)          # (N, J)
    ke = 0.5 * np.sum(speed2, axis=1)          # (N,)

    ke_min = float(np.nanmin(ke))
    ke_max = float(np.nanmax(ke))
    if ke_max > ke_min:
        ke_norm = (ke - ke_min) / (ke_max - ke_min)
    else:
        ke_norm = np.zeros_like(ke)
    return ke_norm.astype(np.float32)


def smooth_signal(x: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Suavizado simple (media móvil). window en nº de frames.
    """
    if window <= 1 or len(x) < window:
        return x.copy()
    w = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, w, mode="same")


def analyze_energy_peaks(ke_norm: np.ndarray) -> Dict[str, Any]:
    """
    Extrae info tipo debug_segmentation:
      - energía suavizada
      - umbral bajo / alto (percentiles)
      - picos por encima del umbral alto
    """
    ke_smooth = smooth_signal(ke_norm, window=15)

    thr_low = float(np.percentile(ke_smooth, 40))
    thr_high = float(np.percentile(ke_smooth, 75))

    if find_peaks is not None:
        peaks, _ = find_peaks(ke_smooth, height=thr_high, distance=10)
        peaks = peaks.astype(int)
    else:
        peaks = np.array([], dtype=int)

    return {
        "ke_smooth": ke_smooth,
        "thr_low": thr_low,
        "thr_high": thr_high,
        "peaks_idx": peaks,
    }


# ======================================================================
# SEGMENTOS DESDE JSON DE MOVIMIENTOS
# ======================================================================

def segments_from_moves_json(moves_json_path: Path) -> List[Dict[str, Any]]:
    """
    Lee tu JSON de movimientos (output de capture_moves_with_spec) y
    devuelve una lista de segmentos con t0, t1, label y score.
    """
    d = json.loads(moves_json_path.read_text(encoding="utf-8"))

    moves = d.get("moves")
    if moves is None:
        moves = d.get("segments", [])

    segments: List[Dict[str, Any]] = []
    for m in moves:
        t0 = float(m.get("t_start", m.get("start_s", 0.0)) or 0.0)
        t1 = float(m.get("t_end", m.get("end_s", t0)) or t0)
        label = (
            str(m.get("M"))
            or str(m.get("label", ""))
            or f"m{m.get('idx', '?')}"
        )
        score = float(m.get("match_score", 0.0) or 0.0)

        segments.append(
            {
                "t0": t0,
                "t1": t1,
                "label": label,
                "score": score,
            }
        )
    return segments


# ======================================================================
# GRÁFICOS
# ======================================================================

def plot_energy_and_segments(
    t: np.ndarray,
    ke_norm: np.ndarray,
    segments: List[Dict[str, Any]],
    title: str,
    out_path: Path,
) -> None:
    """
    Gráfico “principal”: energía cinética global + bandas de segmentos.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(t, ke_norm, label="Energía cinética normalizada")
    ax.set_title(title)
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Energía cinética normalizada")
    ax.set_ylim(0.0, 1.2)

    y1, y2 = 1.02, 1.18
    for seg in segments:
        t0, t1 = seg["t0"], seg["t1"]
        label = seg["label"]
        score = seg["score"]

        if score >= 0.80:
            color = "tab:green"
        elif score >= 0.50:
            color = "tab:orange"
        else:
            color = "tab:red"

        ax.axvspan(t0, t1, alpha=0.12, color=color)
        xm = 0.5 * (t0 + t1)
        # Para evitar saturar, solo mostramos etiqueta si el segmento
        # dura al menos 0.3 s
        if (t1 - t0) >= 0.3:
            ax.text(
                xm,
                y2,
                f"{label}",
                ha="center",
                va="top",
                fontsize=6,
                rotation=90,
            )

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_energy_debug(
    t: np.ndarray,
    ke_norm: np.ndarray,
    energy_info: Dict[str, Any],
    segments: List[Dict[str, Any]],
    sample_id: str,
    out_path: Path,
) -> None:
    """
    Versión más completa inspirada en debug_segmentation:
      - energía global y suavizada
      - umbrales bajo/alto
      - picos
      - histograma de duración de segmentos
      - distribución de “calidad” de segmentos (según score)
    """
    ke_smooth = energy_info["ke_smooth"]
    thr_low = energy_info["thr_low"]
    thr_high = energy_info["thr_high"]
    peaks_idx = energy_info["peaks_idx"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    time = t

    # ---------- Panel 1: energía global ----------
    ax1 = axes[0]
    ax1.plot(time, ke_norm, label="Energía global", alpha=0.6)
    ax1.plot(time, ke_smooth, label="Energía suavizada", linewidth=2)

    for seg in segments:
        ax1.axvspan(seg["t0"], seg["t1"], alpha=0.15, color="tab:green")

    ax1.set_title(f"Energía cinética global - {sample_id}")
    ax1.set_ylabel("Energía (u. relativa)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---------- Panel 2: umbrales + picos ----------
    ax2 = axes[1]
    ax2.plot(time, ke_smooth, label="Energía suavizada", color="black", linewidth=1.8)
    ax2.axhline(thr_low, color="orange", linestyle="--", label=f"Umbral bajo {thr_low:.3f}")
    ax2.axhline(thr_high, color="red", linestyle="--", label=f"Umbral alto {thr_high:.3f}")

    if len(peaks_idx):
        ax2.plot(time[peaks_idx], ke_smooth[peaks_idx], "ro", markersize=5, label="Picos")

    for seg in segments:
        ax2.axvspan(seg["t0"], seg["t1"], alpha=0.18, color="tab:green")

    ax2.set_title("Algoritmo de segmentación por picos de energía")
    ax2.set_ylabel("Energía normalizada")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---------- Panel 3: estadísticas de segmentos ----------
    ax3 = axes[2]

    if segments:
        durations = np.array([s["t1"] - s["t0"] for s in segments], dtype=np.float32)
        scores = np.array([s["score"] for s in segments], dtype=np.float32)

        # Histograma de duraciones
        ax3.hist(
            durations,
            bins=min(10, len(durations)),
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            label="Duraciones",
        )
        mean_dur = float(durations.mean())
        ax3.axhline(
            0,
            color="none",
        )  # para que matplotlib no se queje si no hay datos
        ax3.axvline(
            mean_dur, color="red", linestyle="--", label=f"Duración media: {mean_dur:.2f}s"
        )

        ax3.set_xlabel("Duración de segmento [s]")
        ax3.set_ylabel("Frecuencia")
        ax3.set_title("Distribución de duraciones de segmentos")

        # Texto adicional con score
        if len(scores):
            txt = (
                f"N segmentos: {len(segments)}  |  "
                f"score medio: {scores.mean():.3f}  |  "
                f"score min/max: {scores.min():.3f}/{scores.max():.3f}"
            )
            ax3.text(
                0.01,
                0.98,
                txt,
                transform=ax3.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "Sin segmentos para esta muestra",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax3.set_axis_off()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# MÉTRICAS POR MUESTRA
# ======================================================================

def compute_sample_metrics(
    t: np.ndarray,
    ke_norm: np.ndarray,
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    ke_mean = float(np.nanmean(ke_norm))
    ke_std = float(np.nanstd(ke_norm))
    ke_max = float(np.nanmax(ke_norm))

    n_segments = len(segments)
    if n_segments:
        seg_durations = np.array([s["t1"] - s["t0"] for s in segments], dtype=np.float32)
        scores = np.array([s["score"] for s in segments], dtype=np.float32)

        score_mean = float(scores.mean())
        score_std = float(scores.std())
        score_min = float(scores.min())
        score_max = float(scores.max())

        dur_mean = float(seg_durations.mean())
        dur_std = float(seg_durations.std())
        dur_min = float(seg_durations.min())
        dur_max = float(seg_durations.max())
    else:
        score_mean = score_std = score_min = score_max = float("nan")
        dur_mean = dur_std = dur_min = dur_max = float("nan")

    return {
        "duration_s": duration,
        "ke_mean": ke_mean,
        "ke_std": ke_std,
        "ke_max": ke_max,
        "n_segments": n_segments,
        "seg_score_mean": score_mean,
        "seg_score_std": score_std,
        "seg_score_min": score_min,
        "seg_score_max": score_max,
        "seg_duration_mean": dur_mean,
        "seg_duration_std": dur_std,
        "seg_duration_min": dur_min,
        "seg_duration_max": dur_max,
    }


def analyze_sample(
    csv_path: Path,
    moves_json: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Analiza un par (CSV, JSON de movimientos) y genera:

      - gráfico energía+segmentos
      - gráfico “debug” (picos, umbrales, histogramas)
      - entrada de métricas agregadas (dict)
    """
    print()
    print(f"=== Analizando {csv_path.stem} ===")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {moves_json}")

    t, xyz = load_landmarks_csv(csv_path)
    ke_norm = compute_kinetic_energy(t, xyz)
    segments = segments_from_moves_json(moves_json)
    energy_info = analyze_energy_peaks(ke_norm)

    metrics = compute_sample_metrics(t, ke_norm, segments)
    sample_id = csv_path.stem

    # Gráfico “simple”
    png_simple = out_dir / f"{sample_id}_energia_segmentos.png"
    plot_energy_and_segments(
        t,
        ke_norm,
        segments,
        title=f"Energía cinética y segmentos - {sample_id}",
        out_path=png_simple,
    )
    print(f"  → Gráfico energía+segmentos: {png_simple}")

    # Gráfico “debug” (algoritmo de segmentación)
    png_debug = out_dir / f"{sample_id}_energia_debug.png"
    plot_energy_debug(
        t,
        ke_norm,
        energy_info,
        segments,
        sample_id=sample_id,
        out_path=png_debug,
    )
    print(f"  → Gráfico debug de segmentación: {png_debug}")

    return metrics


# ======================================================================
# ANÁLISIS DE CARPETA COMPLETA
# ======================================================================

def analyze_folder(
    csv_dir: Path,
    moves_dir: Path,
    out_dir: Path,
    pattern: str = "8yang_*.csv",
) -> None:
    """
    Recorre todos los CSV de `csv_dir` (según patrón) y los empareja con
    su JSON de movimientos en `moves_dir`.

    Para cada muestra:
      - genera dos PNG (energía+segmentos y debug)
      - añade una fila a un CSV de métricas.

    CSV de salida: out_dir / 'resumen_metricas.csv'
    """
    csv_files = sorted(csv_dir.glob(pattern))
    if not csv_files:
        print(f"[ERROR] No se encontraron CSV en {csv_dir} con patrón '{pattern}'")
        return

    all_metrics: List[Dict[str, Any]] = []

    for csv_path in csv_files:
        sample_id = csv_path.stem  # ej: 8yang_001

        # Ruta “obvia”
        json_path = moves_dir / f"{sample_id}_moves.json"
        if not json_path.exists():
            # Intentar formas alternativas, p.ej. 8yang_8yang_001_moves.json
            alt = list(moves_dir.glob(f"*{sample_id}_moves.json"))
            if alt:
                json_path = alt[0]

        if not json_path.exists():
            print(
                f"[WARN] No se encontró JSON de movimientos para {sample_id} "
                f"en {moves_dir}, se omite."
            )
            continue

        metrics = analyze_sample(csv_path, json_path, out_dir)
        metrics["sample_id"] = sample_id
        all_metrics.append(metrics)

    if not all_metrics:
        print("[WARN] No se pudo analizar ninguna muestra (faltan JSON).")
        return

    df_metrics = pd.DataFrame(all_metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "resumen_metricas.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\n[OK] Resumen de métricas guardado en {metrics_csv}")

    # Resumen global rápido en consola
    print("\n=== RESUMEN GLOBAL ===")
    print(df_metrics.describe(include="all").transpose())


# ======================================================================
# CLI
# ======================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Análisis de métricas de poomsae a partir de landmarks 3D "
            "y JSON de movimientos (segmentación por energía cinética)."
        )
    )

    # Modo carpeta (recomendado)
    ap.add_argument(
        "--csv-dir",
        type=str,
        default="",
        help="Carpeta con CSV de landmarks (ej: data/landmarks/8yang).",
    )
    ap.add_argument(
        "--moves-dir",
        type=str,
        default="",
        help="Carpeta con JSON de movimientos (ej: data/moves_ml).",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="8yang_*.csv",
        help="Patrón glob para listar CSV (por defecto: 8yang_*.csv).",
    )

    # Modo una sola muestra
    ap.add_argument(
        "--csv",
        type=str,
        default="",
        help="(Opcional) CSV específico de landmarks para analizar una sola muestra.",
    )
    ap.add_argument(
        "--moves-json",
        type=str,
        default="",
        help="(Opcional) JSON específico de movimientos para analizar una sola muestra.",
    )

    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "reports" / "analisis"),
        help="Carpeta de salida para gráficos y resumen CSV.",
    )

    return ap


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    ap = build_argparser()
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)

    # Modo una muestra
    if args.csv and args.moves_json:
        csv_path = Path(args.csv)
        json_path = Path(args.moves_json)
        out_dir.mkdir(parents=True, exist_ok=True)
        _ = analyze_sample(csv_path, json_path, out_dir)
        return

    # Modo carpeta
    if not args.csv_dir or not args.moves_dir:
        ap.error("Debes indicar --csv-dir y --moves-dir o bien --csv y --moves-json.")

    csv_dir = Path(args.csv_dir)
    moves_dir = Path(args.moves_dir)
    analyze_folder(csv_dir, moves_dir, out_dir, pattern=args.pattern)


if __name__ == "__main__":
    main()
