# src/ml/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------
# Utilidades para leer dataset y seleccionar columnas
# ---------------------------------------------------------------------

def _read_dataset(path: Path) -> pd.DataFrame:
    """
    Lee un dataset de entrenamiento a partir de un archivo CSV o Parquet.

    - .parquet / .pq → pd.read_parquet
    - .csv          → pd.read_csv
    """
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"[ERROR] Formato de dataset no soportado: {path}")


def _detect_target_col(df: pd.DataFrame) -> str:
    """
    Detecta la columna de etiqueta si el usuario no la especifica.

    Intenta en este orden: y, label, target, gt_label.
    """
    candidates = ["y", "label", "target", "gt_label"]
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        "[ERROR] No se encontró la columna de etiqueta. "
        "Pasa --target-col explícitamente o crea una columna 'y'/'label'/'target'."
    )


def _select_feature_cols(
    df: pd.DataFrame,
    target_col: str,
    drop_like: List[str] | None = None,
) -> List[str]:
    """
    Selecciona columnas numéricas para usar como features.

    - Excluye la columna de etiqueta.
    - Excluye columnas típicas de metadatos (video_id, tiempos, etc.) si son texto.
    """
    if drop_like is None:
        drop_like = ["video", "video_id", "start_s", "end_s", "move_id", "segment_id"]

    bad = {target_col}
    for c in df.columns:
        if c == target_col:
            continue
        lc = c.lower()
        # Si el nombre indica que podría ser texto de metadatos y el dtype es object, se descarta
        if any(lc.startswith(p) or lc == p for p in drop_like) and df[c].dtype == "O":
            bad.add(c)

    # Solo columnas numéricas
    num_cols: List[str] = [
        c
        for c in df.columns
        if c not in bad and np.issubdtype(df[c].dtype, np.number)
    ]

    if not num_cols:
        raise SystemExit(
            "[ERROR] No se encontraron columnas numéricas de características.\n"
            "Asegúrate de que el dataset tenga features numéricas además de la etiqueta."
        )
    return num_cols


# ---------------------------------------------------------------------
# Construcción de modelo + pipeline
# ---------------------------------------------------------------------

def _make_model(model_name: str, args, n_classes: int):
    """
    Crea el modelo sklearn y define si requiere bloque de escalado.
    Devuelve: (clf, scaler_block)
    """
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight=(args.class_weight if args.class_weight != "none" else None),
            n_jobs=-1,
            random_state=args.seed,
        )
        scaler_block = None

    elif model_name == "hgb":
        clf = HistGradientBoostingClassifier(
            max_depth=args.hgb_max_depth if args.hgb_max_depth > 0 else None,
            learning_rate=args.hgb_lr,
            max_iter=args.hgb_max_iter,
            l2_regularization=args.hgb_l2,
            early_stopping=True,
            random_state=args.seed,
        )
        scaler_block = None

    elif model_name == "svm":
        clf = SVC(
            kernel="rbf",
            C=args.svm_C,
            gamma="scale",
            probability=True,
            class_weight=(args.class_weight if args.class_weight != "none" else None),
            random_state=args.seed,
        )
        scaler_block = (
            "scale",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", RobustScaler()),
                ]
            ),
        )

    elif model_name == "logreg":
        clf = LogisticRegression(
            C=args.lr_C,
            max_iter=2000,
            class_weight=(args.class_weight if args.class_weight != "none" else None),
            n_jobs=-1,
            random_state=args.seed,
        )
        scaler_block = (
            "scale",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                ]
            ),
        )

    else:
        raise SystemExit(f"[ERROR] Modelo no soportado: {model_name}")

    return clf, scaler_block


def _build_pipeline(
    num_cols: List[str],
    model,
    scaler_block,
) -> Pipeline:
    """
    Construye el pipeline:
      - ColumnTransformer para imputación / escalado en columnas numéricas
      - Clasificador final
    """
    if scaler_block is None:
        # Solo imputación mediana
        preproc = ColumnTransformer(
            transformers=[("num", SimpleImputer(strategy="median"), num_cols)],
            remainder="drop",
        )
    else:
        # scaler_block = ("scale", Pipeline([...]))
        preproc = ColumnTransformer(
            transformers=[("num", scaler_block[1], num_cols)],
            remainder="drop",
        )

    pipe = Pipeline(
        [
            ("pre", preproc),
            ("clf", model),
        ]
    )
    return pipe


# ---------------------------------------------------------------------
# Validación cruzada (CV) estratificada
# ---------------------------------------------------------------------

def _kfold_eval(
    pipe: Pipeline,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    cv_splits: int,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> Tuple[float, float, float, float]:
    """
    Ejecuta validación cruzada estratificada y devuelve:
    - F1_macro medio y desviación estándar
    - Balanced Accuracy medio y desviación estándar
    """
    skf = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=seed,
    )
    f1s: List[float] = []
    bals: List[float] = []

    for tr_idx, va_idx in skf.split(X, y_enc):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y_enc[tr_idx], y_enc[va_idx]

        fit_kwargs = {}
        if sample_weight is not None:
            # sample_weight se pasa al estimador dentro del pipeline
            fit_kwargs["clf__sample_weight"] = sample_weight[tr_idx]

        pipe.fit(Xtr, ytr, **fit_kwargs)
        yhat = pipe.predict(Xva)

        f1s.append(f1_score(yva, yhat, average="macro"))
        bals.append(balanced_accuracy_score(yva, yhat))

    return (
        float(np.mean(f1s)),
        float(np.std(f1s)),
        float(np.mean(bals)),
        float(np.std(bals)),
    )


# ---------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Entrena y guarda un clasificador de calidad/etiqueta de movimiento "
            "(por ejemplo posturas de Poomsae) usando un dataset de features."
        )
    )
    ap.add_argument(
        "--data",
        required=True,
        help="Ruta .parquet / .pq / .csv con características + columna de etiqueta.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Ruta .joblib de salida (se crea también un .metrics.json).",
    )
    ap.add_argument(
        "--model",
        default="rf",
        choices=["rf", "hgb", "svm", "logreg"],
        help="Tipo de modelo base (default: rf).",
    )
    ap.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Número de folds para cross-validation estratificada (default: 5).",
    )

    # General
    ap.add_argument(
        "--class-weight",
        default="none",
        choices=["none", "balanced"],
        help="Esquema de pesos de clase (none / balanced).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )
    ap.add_argument(
        "--target-col",
        default="",
        help=(
            "Nombre de la columna de etiqueta. "
            "Si se omite, se intenta detectar (y/label/target/gt_label)."
        ),
    )
    ap.add_argument(
        "--drop-cols",
        nargs="*",
        default=None,
        help=(
            "Lista de prefijos/nombres de columnas a descartar como features "
            "(además de la etiqueta). Útil para eliminar metadatos."
        ),
    )

    # RF
    ap.add_argument("--rf-n-estimators", type=int, default=400)
    ap.add_argument("--rf-max-depth", type=int, default=20)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)

    # HGB
    ap.add_argument("--hgb-max-depth", type=int, default=8)
    ap.add_argument("--hgb-max-iter", type=int, default=500)
    ap.add_argument("--hgb-lr", type=float, default=0.05)
    ap.add_argument("--hgb-l2", type=float, default=0.0)

    # SVM
    ap.add_argument("--svm-C", type=float, default=4.0)

    # LogReg
    ap.add_argument("--lr-C", type=float, default=2.0)

    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    # 1) Cargar dataset
    df = _read_dataset(data_path)

    # 2) Determinar columna de etiqueta
    if args.target_col:
        if args.target_col not in df.columns:
            raise SystemExit(
                f"[ERROR] La columna de etiqueta '{args.target_col}' no está en el dataset."
            )
        target_col = args.target_col
    else:
        target_col = _detect_target_col(df)

    # 3) Codificar etiquetas a enteros
    y_raw = df[target_col].astype(str).values
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)

    # 4) Selección de columnas de características
    feat_cols = _select_feature_cols(
        df,
        target_col=target_col,
        drop_like=args.drop_cols,
    )
    X = df[feat_cols].copy()

    # 5) Info de clases
    classes = list(le.classes_)
    print("[INFO] Clases encontradas:", classes)
    cls_counts = pd.Series(y).value_counts().sort_index()
    print(
        "[INFO] Distribución por clase:",
        {classes[i]: int(cls_counts.get(i, 0)) for i in range(len(classes))},
    )
    print(f"[INFO] Nº ejemplos: {len(df)}, Nº features: {len(feat_cols)}")

    # 6) Modelo + preprocesamiento
    clf, scaler_block = _make_model(args.model, args, n_classes=len(classes))
    pipe = _build_pipeline(feat_cols, clf, scaler_block)

    # 7) sample_weight para modelos sin class_weight nativo (p.ej. HGB)
    sample_weight = None
    if args.model == "hgb" and args.class_weight == "balanced":
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(classes)),
            y=y,
        )
        wmap = {i: weights[i] for i in range(len(classes))}
        sample_weight = np.array([wmap[yi] for yi in y], dtype=np.float32)

    # 8) Validación cruzada
    print("\n[INFO] Ejecutando cross-validation estratificada...")
    f1m, f1s, bam, bas = _kfold_eval(
        pipe,
        X,
        y,
        args.cv,
        args.seed,
        sample_weight=sample_weight,
    )
    print(
        f"[CV] F1_macro: {f1m:.4f} ± {f1s:.4f} | "
        f"BalancedAcc: {bam:.4f} ± {bas:.4f}"
    )

    # 9) Entrenamiento final en todo el conjunto
    print("\n[INFO] Entrenando modelo final sobre todo el dataset...")
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight

    pipe.fit(X, y, **fit_kwargs)

    # 10) Reporte sobre el conjunto completo (referencial, puede estar sobreajustado)
    yhat = pipe.predict(X)
    print("\n[TRAIN] Classification report (referencial, mismo dataset de entrenamiento):")
    print(classification_report(y, yhat, target_names=classes, digits=3))
    print("[TRAIN] Matriz de confusión:\n", confusion_matrix(y, yhat))

    # 11) Guardar modelo + metadatos para uso en StanceClassifier
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": args.model,
        "cv_f1_macro_mean": f1m,
        "cv_f1_macro_std": f1s,
        "cv_balanced_acc_mean": bam,
        "cv_balanced_acc_std": bas,
        "classes": classes,
        "data_path": str(data_path),
        "target_col": target_col,
        "feature_cols": feat_cols,
        "n_samples": int(len(df)),
        "n_features": int(len(feat_cols)),
        "class_weight_mode": args.class_weight,
        "seed": int(args.seed),
    }
    payload = {
        "model": pipe,
        "feature_cols": feat_cols,
        "label_encoder": le,
        "meta": meta,
    }
    joblib.dump(payload, out_path)

    # 12) Métricas a JSON (útil para monitorear entrenamientos)
    metrics_json = out_path.with_suffix(".metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Modelo guardado en   -> {out_path}")
    print(f"[OK] Métricas guardadas en -> {metrics_json}")


if __name__ == "__main__":
    main()
