#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slowdown_video.py
-----------------
Herramienta para cambiar velocidad de reproducción de videos.

Modos:

  1) --mode fps
     - Cambia solo los FPS del archivo (rápido, pero no añade fluidez).
     - Ejemplo: pasar de 60 fps -> 30 fps para ver más lento.
  
  2) --mode interp
     - Mantiene el FPS original.
     - Genera cámara lenta creando frames extra entre cada par de frames.
     - Nuevo parámetro:
         --interp-method duplicate  (por defecto, más nítido)
         --interp-method blend      (más fluido/cinemático, pero borroso)

Ejemplos:

  # Cámara lenta nítida (duplica frames, duración x2 aprox)
  python -m src.tools.slowdown_video --mode interp --slow-factor 2.0 \
         --interp-method duplicate cam_3.mp4

  # Cámara lenta con blending (más blur, pero muy suave)
  python -m src.tools.slowdown_video --mode interp --slow-factor 2.0 \
         --interp-method blend cam_3.mp4

  # Solo cambiar FPS (rápido, sin interpolar)
  python -m src.tools.slowdown_video --mode fps --speed 0.5 cam_3.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np


# --------------------------------------------------------------------
# MODO 1: Cambiar solo FPS (rápido, sin añadir fluidez)
# --------------------------------------------------------------------
def slow_down_video_change_fps(
    input_path: Path,
    output_path: Path,
    speed: float = 0.5,
    codec: str = "mp4v",
) -> None:
    """
    Re-encodea el video cambiando solo su FPS.
    speed < 1.0  -> más lento (ej: 0.5 = mitad de velocidad)
    speed > 1.0  -> más rápido
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERR] No se pudo abrir: {input_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_fps = max(1.0, float(orig_fps) * float(speed))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, new_fps, (width, height))

    if not writer.isOpened():
        print(f"[ERR] No se pudo crear el writer para: {output_path}")
        cap.release()
        return

    print(
        f"[FPS] {input_path.name}: {orig_fps:.2f} FPS -> {new_fps:.2f} FPS "
        f"(speed={speed}) frames={n_frames}"
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  ... {frame_idx}/{n_frames} frames", end="\r")

    cap.release()
    writer.release()
    print(f"\n[OK] Guardado (modo FPS): {output_path}\n")


# --------------------------------------------------------------------
# MODO 2: Interpolación de frames (cámara lenta)
# --------------------------------------------------------------------
def slow_down_video_interpolated(
    input_path: Path,
    output_path: Path,
    slow_factor: float = 2.0,
    codec: str = "mp4v",
    interp_method: str = "duplicate",  # 'duplicate' o 'blend'
) -> None:
    """
    Genera cámara lenta manteniendo el FPS original.

    - slow_factor ≈ 1.0  -> casi igual duración
    - slow_factor = 2.0  -> el video dura ~2x (50% velocidad)
    - slow_factor = 3.0  -> el video dura ~3x, etc.

    interp_method:
      - 'duplicate': añade frames duplicados (más nítido, menos blur).
      - 'blend'    : añade frames interpolados por mezcla lineal
                     (más suave/cinemático, pero puede verse borroso).
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERR] No se pudo abrir: {input_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_frames <= 0:
        print(f"[ERR] No se pudo obtener CAP_PROP_FRAME_COUNT para: {input_path}")
        cap.release()
        return

    # FPS de salida = FPS original (se percibe igual de fluido)
    out_fps = orig_fps

    # Número aproximado de frames extra entre cada par:
    #   slow_factor ≈ 1 + add_per_pair  -> dure ~slow_factor veces más
    add_per_pair = max(1, int(round(slow_factor - 1.0)))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

    if not writer.isOpened():
        print(f"[ERR] No se pudo crear el writer para: {output_path}")
        cap.release()
        return

    print(
        f"[INTERP-{interp_method}] {input_path.name}: {orig_fps:.2f} FPS, "
        f"frames_in={n_frames}, slow_factor≈{slow_factor:.2f} "
        f"(add_per_pair={add_per_pair})"
    )

    # --- Pre-lectura del primer frame ---
    ok, prev = cap.read()
    if not ok:
        print(f"[ERR] No se pudo leer el primer frame de: {input_path}")
        cap.release()
        writer.release()
        return

    total_written = 0

    while True:
        ok, nxt = cap.read()
        if not ok:
            # Último frame: lo escribimos una vez más
            writer.write(prev)
            total_written += 1
            break

        # 1) Escribimos el frame original (prev)
        writer.write(prev)
        total_written += 1

        # 2) Insertamos frames extras entre prev y nxt
        for k in range(1, add_per_pair + 1):
            if interp_method == "blend":
                # α crece de 1/(add+1) a add/(add+1)
                alpha = float(k) / float(add_per_pair + 1)
                inter = cv2.addWeighted(prev, 1.0 - alpha, nxt, alpha, 0.0)
            else:
                # 'duplicate': mantenemos el frame previo (casi nada de blur)
                inter = prev

            writer.write(inter)
            total_written += 1

        prev = nxt

    cap.release()
    writer.release()
    print(f"[OK] Guardado (modo interp-{interp_method}): {output_path}")
    print(f"    Frames entrada: {n_frames}, frames salida: {total_written}\n")


# --------------------------------------------------------------------
# Utilidades CLI
# --------------------------------------------------------------------
def iter_input_videos(args: argparse.Namespace) -> Iterable[Path]:
    """Devuelve la lista de videos a procesar según CLI."""
    videos: List[Path] = []

    # 1) Videos explícitos (posicionales)
    for v in args.videos or []:
        p = Path(v)
        if p.exists():
            videos.append(p)
        else:
            print(f"[WARN] Ignorando (no existe): {p}")

    # 2) Carpeta + patrón
    if args.input_dir:
        base = Path(args.input_dir)
        if not base.exists():
            print(f"[WARN] Carpeta no existe: {base}")
        else:
            for p in sorted(base.glob(args.pattern)):
                if p.is_file():
                    videos.append(p)

    # Eliminar duplicados manteniendo orden
    seen = set()
    unique: List[Path] = []
    for p in videos:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cambiar velocidad de videos (FPS o interpolación)."
    )
    p.add_argument(
        "videos",
        nargs="*",
        help="Rutas de videos a procesar (opcional si usas --input-dir).",
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Carpeta con videos a procesar (opcional).",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="Patrón glob para --input-dir (por defecto: '*.mp4').",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["fps", "interp"],
        default="interp",
        help="Modo: 'fps' = solo cambiar FPS, 'interp' = cámara lenta.",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Factor de velocidad para mode=fps (1.0=igual, 0.5=mitad, 2.0=doble).",
    )
    p.add_argument(
        "--slow-factor",
        type=float,
        default=2.0,
        help="Factor de alargamiento para mode=interp (≈duración x slow-factor).",
    )
    p.add_argument(
        "--interp-method",
        type=str,
        choices=["duplicate", "blend"],
        default="duplicate",
        help="Método de interpolación para mode=interp: "
             "'duplicate' (más nítido) o 'blend' (más suave, con blur).",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Sufijo para el nombre de salida (si vacío, se genera automático).",
    )
    p.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC de códec (mp4v, XVID, avc1, etc.).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    videos = list(iter_input_videos(args))
    if not videos:
        parser.error("No se encontraron videos. Usa rutas posicionales o --input-dir.")

    if args.mode == "fps":
        speed = float(args.speed)
        if speed <= 0:
            parser.error("--speed debe ser > 0 (ej. 0.5 para más lento).")
    else:
        slow_factor = float(args.slow_factor)
        if slow_factor <= 0:
            parser.error("--slow-factor debe ser > 0 (ej. 2.0 para 2x más lento).")

    for vin in videos:
        # Construir nombre de salida
        if args.suffix:
            suf = args.suffix
        else:
            if args.mode == "fps":
                suf = f"_fps{int(args.speed*100):03d}"
            else:
                suf = f"_slowx{args.slow_factor:.1f}_{args.interp_method}"
                suf = suf.replace(".", "p")

        stem = vin.stem
        out_name = f"{stem}{suf}{vin.suffix}"
        vout = vin.with_name(out_name)

        if args.mode == "fps":
            slow_down_video_change_fps(
                input_path=vin,
                output_path=vout,
                speed=args.speed,
                codec=args.codec,
            )
        else:
            slow_down_video_interpolated(
                input_path=vin,
                output_path=vout,
                slow_factor=args.slow_factor,
                codec=args.codec,
                interp_method=args.interp_method,
            )


if __name__ == "__main__":
    main()
