# src/tools/debug_move_capture_rt.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import sys

# ---------------------------------------------------------------------
# AÑADIR LA RAÍZ DEL PROYECTO A sys.path
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../poomsae_kinect_3d
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ahora se puede hacer import "src.segmentation.move_capture"
from src.segmentation.move_capture import (
    PoomsaeConfig,
    SpecAwareSegmenter,
    load_landmarks_csv,
    series_xy,
    LMK,
    angle3,
)


# ---------------------------------------------------------------------
# Helpers: construir landmarks_dict y angles_dict
# ---------------------------------------------------------------------
def build_landmarks_dict(df: pd.DataFrame, nframes: int) -> Dict[str, np.ndarray]:
    needed_points = [
        "L_SH", "R_SH", "L_HIP", "R_HIP",
        "L_WRIST", "R_WRIST", "L_ANK", "R_ANK",
        "L_KNEE", "R_KNEE", "L_ELB", "R_ELB",
        "L_HEEL", "R_HEEL", "L_FOOT", "R_FOOT",
    ]
    out: Dict[str, np.ndarray] = {}
    for name in needed_points:
        idx = LMK.get(name)
        if idx is None:
            continue
        out[name] = series_xy(df, idx, nframes)
    return out


def build_angles_dict(ld: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    angles: Dict[str, np.ndarray] = {}

    try:
        # Rodillas
        l_hip = ld["L_HIP"]; r_hip = ld["R_HIP"]
        l_knee = ld["L_KNEE"]; r_knee = ld["R_KNEE"]
        l_ank = ld["L_ANK"]; r_ank = ld["R_ANK"]
        min_len = min(map(len, [l_hip, r_hip, l_knee, r_knee, l_ank, r_ank]))

        left_knee = []
        right_knee = []
        for i in range(min_len):
            left_knee.append(angle3(l_hip[i], l_knee[i], l_ank[i]))
            right_knee.append(angle3(r_hip[i], r_knee[i], r_ank[i]))
        angles["left_knee"] = np.array(left_knee, np.float32)
        angles["right_knee"] = np.array(right_knee, np.float32)

        # Codos
        l_sh = ld["L_SH"]; r_sh = ld["R_SH"]
        l_elb = ld["L_ELB"]; r_elb = ld["R_ELB"]
        l_wri = ld["L_WRIST"]; r_wri = ld["R_WRIST"]
        min_len_elb = min(map(len, [l_sh, r_sh, l_elb, r_elb, l_wri, r_wri]))

        left_elb = []
        right_elb = []
        for i in range(min_len_elb):
            left_elb.append(angle3(l_sh[i], l_elb[i], l_wri[i]))
            right_elb.append(angle3(r_sh[i], r_elb[i], r_wri[i]))
        angles["left_elbow"] = np.array(left_elb, np.float32)
        angles["right_elbow"] = np.array(right_elb, np.float32)

        # Caderas
        left_hip_ang = []
        right_hip_ang = []
        for i in range(min_len):
            left_hip_ang.append(angle3(l_sh[i], l_hip[i], l_knee[i]))
            right_hip_ang.append(angle3(r_sh[i], r_hip[i], r_knee[i]))
        angles["left_hip"] = np.array(left_hip_ang, np.float32)
        angles["right_hip"] = np.array(right_hip_ang, np.float32)

    except Exception as e:
        print(f"[DEBUG_RT] Error calculando ángulos: {e}")

    return angles


# ---------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="debug_move_capture_rt",
        description=(
            "Debug visual en tiempo real de la segmentación move_capture "
            "(energía poomsae-aware + segmentos sobre el video)."
        ),
    )
    ap.add_argument(
        "csv",
        type=Path,
        help="CSV de landmarks (ej. data/landmarks/8yang/8yang_001.csv)",
    )
    ap.add_argument(
        "video",
        type=Path,
        help="Video MP4 original (ej. data/raw_videos/8yang/train/8yang_006.mp4)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="YAML de segmentación (default=config/default.yaml)",
    )
    ap.add_argument(
        "--spec",
        type=Path,
        default=Path("config/patterns/8yang_spec.json"),
        help="JSON de spec 8yang (default=config/patterns/8yang_spec.json)",
    )
    ap.add_argument(
        "--pose-spec",
        type=Path,
        default=Path("config/patterns/pose_spec.json"),
        help="JSON de tolerancias de pose (default=config/patterns/pose_spec.json)",
    )
    ap.add_argument(
        "--fps-override",
        type=float,
        default=None,
        help="Si se indica, se usa este FPS en vez del del video.",
    )
    ap.add_argument(
        "--save-video",
        action="store_true",
        help="Si se activa, guarda el vídeo con overlay de energía y segmentos.",
    )
    ap.add_argument(
        "--out-video",
        type=Path,
        default=None,
        help=(
            "Ruta de salida del MP4 de debug. "
            "Si es una carpeta, se genera un nombre automático dentro."
        ),
    )
    return ap


# ---------------------------------------------------------------------
# Lógica principal
# ---------------------------------------------------------------------
def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    csv_path: Path = args.csv
    vid_path: Path = args.video
    cfg_path: Path = args.config
    spec_path: Path = args.spec
    pose_spec_path: Path = args.pose_spec
    fps_override: float | None = args.fps_override
    save_video: bool = args.save_video or (args.out_video is not None)
    out_video_arg: Path | None = args.out_video

    if not csv_path.exists():
        print(f"❌ CSV no encontrado: {csv_path}")
        return
    if not vid_path.exists():
        print(f"❌ Video no encontrado: {vid_path}")
        return

    # 1) Cargar landmarks
    df = load_landmarks_csv(csv_path)
    nframes = int(df["frame"].max()) + 1

    cap = cv2.VideoCapture(str(vid_path))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps_override is not None and fps_override > 0:
        fps = float(fps_override)
        print(f"[DEBUG_RT] Usando FPS override: {fps:.2f}")
    else:
        fps = float(fps_video) if fps_video > 0 else 30.0
        print(f"[DEBUG_RT] FPS leídos del video: {fps:.2f} (fallback 30.0 si 0)")

    print(f"[DEBUG_RT] CSV={csv_path.name}  VIDEO={vid_path.name}  frames={nframes}, fps={fps:.1f}")

    # 2) Config y segmentador
    config = PoomsaeConfig(
        spec_path=spec_path,
        pose_spec_path=pose_spec_path,
        config_path=cfg_path,
    )

    landmarks_dict = build_landmarks_dict(df, nframes)
    angles_dict = build_angles_dict(landmarks_dict)

    segmenter = SpecAwareSegmenter(config)

    # 3) Energía y segmentos
    energy = segmenter._compute_poomsae_aware_energy(
        angles_dict, landmarks_dict, fps
    )
    if energy.size == 0:
        print("[DEBUG_RT] Energía vacía (revisar CSV / landmarks)")
        return

    # Normalización + 'exagerar' visualmente para que se vea más la barra
    energy_norm = energy / (np.max(energy) + 1e-6)

    segments = segmenter.find_segments(angles_dict, landmarks_dict, fps)

    seg_map = np.zeros(nframes, dtype=int)
    for i, (a, b) in enumerate(segments, start=1):
        a = max(0, min(a, nframes - 1))
        b = max(0, min(b, nframes - 1))
        seg_map[a:b + 1] = i

    print(f"[DEBUG_RT] Segmentos detectados: {len(segments)}")
    if len(segments) < 20:
        print("⚠️  Advertencia: pocos segmentos para un 8yang (esperados ~36).")

    # 4) Configurar writer si se va a guardar video
    writer = None
    out_path: Path | None = None

    def _init_writer_if_needed(frame: np.ndarray):
        nonlocal writer, out_path
        if writer is not None or not save_video:
            return

        h, w = frame.shape[:2]

        # Decidir ruta de salida
        if out_video_arg is None:
            out_dir = ROOT / "debug_videos"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"debug_{csv_path.stem}_{vid_path.stem}.mp4"
        else:
            if out_video_arg.suffix == "":
                # Es carpeta
                out_dir = out_video_arg
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"debug_{csv_path.stem}_{vid_path.stem}.mp4"
            else:
                # Es archivo completo
                out_path = out_video_arg
                out_path.parent.mkdir(parents=True, exist_ok=True)

        # ⚠️ IMPORTANTE: NO usar avc1 para evitar OpenH264
        candidates = ["mp4v", "XVID", "MJPG"]

        for cc in candidates:
            fourcc = cv2.VideoWriter_fourcc(*cc)
            wtr = cv2.VideoWriter(
                str(out_path),
                fourcc,
                float(fps),
                (w, h),
                True,
            )
            if wtr is not None and wtr.isOpened():
                writer = wtr
                print(f"[DEBUG_RT] Grabando video de debug en: {out_path} (codec={cc})")
                break

        if writer is None:
            print("[DEBUG_RT] ❌ No se pudo abrir VideoWriter, no se guardará video.")
            out_path = None
    # 5) Repro con overlay
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= nframes:
            break

        # Inicializar writer en el primer frame válido
        _init_writer_if_needed(frame)

        t = frame_idx / fps
        e_val = float(energy_norm[frame_idx]) if frame_idx < len(energy_norm) else 0.0

        # Para debug visual, realzamos energías bajas
        e_vis = float(np.sqrt(max(e_val, 0.0)))  # raíz cuadrada para que se note más
        seg_id = int(seg_map[frame_idx])

        h, w = frame.shape[:2]

        # --- Zona de barra: fondo negro para que destaque ---
        cv2.rectangle(frame, (10, h - 50), (w - 10, h - 10), (0, 0, 0), -1)

        # Longitud de la barra (mínimo 1px si hay algo de energía)
        bar_max_width = (w - 40)
        bar_len = int(e_vis * bar_max_width)
        if e_val > 0 and bar_len < 2:
            bar_len = 2

        # Barra verde + contorno blanco
        cv2.rectangle(frame, (20, h - 40), (20 + bar_len, h - 20), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, h - 40), (w - 20, h - 20), (255, 255, 255), 1)

        txt1 = f"frame {frame_idx}/{nframes-1}  t={t:5.2f}s"
        txt2 = f"energy={e_val:0.3f}  segment={seg_id if seg_id>0 else '-'}"

        cv2.putText(frame, txt1, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, txt2, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Mostrar en pantalla
        cv2.imshow("Debug move_capture (energia y segmentos)", frame)

        # Guardar en video si está activo el writer
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == 27:  # ESC
            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[DEBUG_RT] Video de debug guardado en: {out_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
