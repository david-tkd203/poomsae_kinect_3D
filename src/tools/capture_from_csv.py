"""CLI mínimo para ejecutar el segmentador `capture_moves_with_spec` sobre un CSV
y guardar el JSON resultante en `data/moves_ml`.

Uso:
  python -m src.tools.capture_from_csv --csv data/landmarks/session_20251127_131140/cam_3.csv \
      --video captures/session_20251127_131140/cam_3.mp4 \
      --out-moves-dir data/moves_ml --sample-id session_20251127_131140_cam_3

El JSON generado incluye `segments` y `summary` (compatible con score_pal_yang.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional
import yaml

from ..segmentation.move_capture import (
    capture_moves_with_spec,
    CaptureResult,
)

from ..main_kinect import _augment_json_for_report


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run capture_moves_with_spec over a landmarks CSV")
    ap.add_argument("--csv", required=True, help="Path to landmarks CSV")
    ap.add_argument("--video", default=None, help="Optional video path")
    ap.add_argument("--config", default=None, help="Optional config YAML")
    ap.add_argument("--spec", default=None, help="Optional poomsae spec JSON")
    ap.add_argument("--pose-spec", default=None, help="Optional pose spec JSON")
    ap.add_argument("--out-moves-dir", default="data/moves_ml", help="Output folder for moves JSON")
    ap.add_argument("--sample-id", default=None, help="Sample id to name the output JSON (ej: session_cam3)")
    ap.add_argument("--use-ml", action="store_true", help="Enable ML classifier if model provided")
    ap.add_argument("--ml-model", default=None, help="Path to ML model (joblib) if --use-ml")
    ap.add_argument("--force-fps", type=float, default=None, help="Force FPS to use for segmentation (overrides video fps)")
    # Segmenter tuning overrides (optional)
    ap.add_argument("--seg-peak-thr", type=float, default=None, help="Override segmentation peak_threshold")
    ap.add_argument("--seg-activity-thr", type=float, default=None, help="Override segmentation activity_threshold")
    ap.add_argument("--seg-min-peak-dist", type=int, default=None, help="Override segmentation min_peak_distance (frames)")
    ap.add_argument("--seg-smooth-win", type=int, default=None, help="Override segmentation smooth_window")
    ap.add_argument("--seg-min-seg-frames", type=int, default=None, help="Override segmentation min_segment_frames")

    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    video_path = Path(args.video) if args.video else None
    out_dir = Path(args.out_moves_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_id = args.sample_id or csv_path.stem

    # Si se pasan overrides para segmentación, crear un config temporal basado en el default
    base_config_path = Path(args.config) if args.config else Path("config/default.yaml")
    config_to_use = base_config_path
    if any([args.seg_peak_thr is not None, args.seg_activity_thr is not None,
            args.seg_min_peak_dist is not None, args.seg_smooth_win is not None,
            args.seg_min_seg_frames is not None]):
        try:
            with open(base_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        seg = cfg.get("segmentation", {})
        if args.seg_peak_thr is not None:
            seg["peak_threshold"] = float(args.seg_peak_thr)
        if args.seg_activity_thr is not None:
            seg["activity_threshold"] = float(args.seg_activity_thr)
        if args.seg_min_peak_dist is not None:
            seg["min_peak_distance"] = int(args.seg_min_peak_dist)
        if args.seg_smooth_win is not None:
            seg["smooth_window"] = int(args.seg_smooth_win)
        if args.seg_min_seg_frames is not None:
            seg["min_segment_frames"] = int(args.seg_min_seg_frames)

        cfg["segmentation"] = seg
        tmp_config = out_dir / f"tmp_config_{sample_id}.yaml"
        with open(tmp_config, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        config_to_use = tmp_config

    result: CaptureResult = capture_moves_with_spec(
        csv_path=csv_path,
        video_path=video_path,
        config_path=config_to_use,
        spec_path=Path(args.spec) if args.spec else Path("config/patterns/8yang_spec.json"),
        pose_spec_path=Path(args.pose_spec) if args.pose_spec else Path("config/patterns/pose_spec.json"),
        use_ml_classifier=bool(args.use_ml),
        ml_model_path=Path(args.ml_model) if args.ml_model else None,
        override_fps=float(args.force_fps) if args.force_fps else None,
    )

    # Use the CSV stem (video filename without extension) as the canonical video_id
    # This ensures the JSON references the actual CSV name (e.g., 'cam_3') so downstream
    # tools can find the landmarks file using `--alias` + video_id.
    video_id = csv_path.stem
    full_dict = _augment_json_for_report(result, video_id=video_id)

    out_json = out_dir / f"{sample_id}_moves.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_dict, f, ensure_ascii=False, indent=2)

    print(f"[OK] Guardado moves JSON -> {out_json}")


if __name__ == "__main__":
    main()
