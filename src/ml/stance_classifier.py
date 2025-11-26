# src/ml/stance_classifier.py
"""
Clasificador ML para posturas de Taekwondo (ap_kubi, dwit_kubi, beom_seogi, etc.).

- En producción se recomienda entrenar usando `src/ml/train.py`:
    python -m src.ml.train --data ... --out models/stance_model.joblib ...

  Ese script guarda un payload joblib con:
    {
        "model": sklearn.Pipeline,
        "feature_cols": [...],
        "label_encoder": LabelEncoder,
        "meta": {...}
    }

- Este módulo proporciona una clase `StanceClassifier` ligera para uso en tiempo real:
    - carga el .joblib entrenado,
    - alinea los features,
    - predice posturas a partir de vectores de características.

También mantiene helpers para entrenamiento rápido desde CSV (formato legacy),
pero el flujo recomendado es usar `train.py`.

Integrado y alineado con poomsae_kinect_3d.
"""

from __future__ import annotations

from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

# Importa nombres de features base desde tu extractor ACTUAL
# Ojo: antes venía de ..features.stance_features; ahora de feature_extractor.py
from ..features.feature_extractor import FEATURE_NAMES as BASE_FEATURES


class StanceClassifier:
    """
    Clasificador de posturas basado en un modelo sklearn (RandomForest o Pipeline).

    Uso recomendado (producción):
    -----------------------------
        from src.ml.stance_classifier import StanceClassifier

        clf = StanceClassifier.load("models/stance_model.joblib")
        x = np.array([[... features en el mismo orden que clf.feature_names ...]],
                     dtype=np.float32)
        stance = clf.predict(x)[0]

    El método `load()` es capaz de:
      - Cargar el formato NUEVO de `train.py` (keys: model, feature_cols, label_encoder, meta)
      - Cargar el formato LEGACY guardado con `save()` de esta misma clase (keys: clf, ...).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_split: int = 5,
        random_state: int = 42,
        class_weight: str | None = "balanced",
    ) -> None:
        """
        Si se usa para entrenamiento rápido 'ad-hoc', se crea un RandomForest
        base. En producción normalmente se usará `load()` con un modelo ya
        entrenado desde `train.py`.

        Args:
            n_estimators:
                Número de árboles en el bosque.
            max_depth:
                Profundidad máxima de cada árbol.
            min_samples_split:
                Mínimo de muestras para hacer un split.
            random_state:
                Semilla para reproducibilidad.
            class_weight:
                Manejo de desbalance de clases. e.g. 'balanced', None.
        """
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            class_weight=class_weight,
        )
        self.label_encoder = LabelEncoder()
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        self.meta: Dict[str, Any] | None = None  # info adicional (opcional)

    # ------------------------------------------------------------------
    # Entrenamiento (legacy / rápido). Para entrenamientos serios usar src/ml/train.py
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Entrena el clasificador con cross-validation (modo 'rápido').

        En proyectos grandes se recomienda usar `src/ml/train.py` en su lugar
        y solo usar este método para experimentos pequeños.

        Args:
            X_train:
                Features de entrenamiento (n_samples, n_features).
            y_train:
                Etiquetas (n_samples,), como strings.
            feature_names:
                Nombres de las features, para imprimir importancias.
            cv_folds:
                Número de folds para cross-validation estratificada.

        Returns:
            Dict con métricas de entrenamiento (cv_mean, cv_std, importances, etc.).
        """
        self.feature_names = feature_names

        # Codificar etiquetas
        y_encoded = self.label_encoder.fit_transform(y_train)

        print("🎓 Entrenando Random Forest para posturas (modo rápido)...")
        print(f"   Muestras: {len(X_train)}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Clases:   {list(self.label_encoder.classes_)}")
        print(
            f"   Distribución: "
            f"{dict(zip(*np.unique(y_train, return_counts=True)))}"
        )

        # Cross-validation
        print(f"\n🔄 Cross-validation ({cv_folds}-fold estratificado)...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.clf, X_train, y_encoded, cv=cv, scoring="accuracy"
        )

        print(
            "   Accuracy por fold: "
            + ", ".join(f"{s:.3f}" for s in cv_scores)
        )
        print(f"   Media: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Entrenamiento final
        self.clf.fit(X_train, y_encoded)
        self.is_fitted = True

        # Feature importance (si tenemos nombres)
        if feature_names is not None:
            importances = self.clf.feature_importances_
            feat_imp = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
            print("\n📊 Feature Importances:")
            for name, imp in feat_imp:
                print(f"   {name:20s}: {imp:.4f}")
        else:
            importances = self.clf.feature_importances_

        return {
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_scores": [float(s) for s in cv_scores],
            "importances": importances.tolist(),
        }

    # ------------------------------------------------------------------
    # Evaluación
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evalúa el clasificador en un conjunto de prueba.

        Args:
            X_test:
                Features de prueba.
            y_test:
                Etiquetas verdaderas (strings).
            verbose:
                Si imprimir métricas legibles.

        Returns:
            Dict con métricas detalladas (accuracy, precision, recall, f1, cm, etc.).
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado. Llamar train() o load() primero.")

        # Predicciones
        y_pred = self.predict(X_test)

        # Métricas globales
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average="weighted"
        )

        # Reporte por clase (dict)
        class_report = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
        )

        # Matriz de confusión
        cm = confusion_matrix(
            y_test, y_pred, labels=self.label_encoder.classes_
        )

        if verbose:
            print("\n📊 EVALUACIÓN EN TEST SET")
            print(f"   Accuracy:  {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            print(f"   F1-score:  {f1:.3f}")

            print("\n📋 Reporte por clase:")
            print(
                classification_report(
                    y_test,
                    y_pred,
                    target_names=self.label_encoder.classes_,
                )
            )

            print("\n🔲 Confusion Matrix:")
            print(cm)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "class_report": class_report,
        }

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice clases para nuevas muestras (devuelve etiquetas string).

        X:
            Array de forma (n_samples, n_features) con las features en el
            MISMO orden que `self.feature_names` (o `BASE_FEATURES` si no está definido).
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado. Usa load() o train().")
        y_encoded = self.clf.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades por clase para cada muestra."""
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado. Usa load() o train().")
        if not hasattr(self.clf, "predict_proba"):
            raise RuntimeError("El modelo actual no soporta predict_proba().")
        return self.clf.predict_proba(X)

    def predict_from_dict(self, feat_dict: Dict[str, float]) -> str:
        """
        Conveniencia: toma un diccionario {feature_name: valor} y
        predice la postura para UN solo ejemplo.

        Se alinea usando `self.feature_names` (si existe) o BASE_FEATURES.
        """
        if self.feature_names is not None:
            names = self.feature_names
        else:
            names = list(BASE_FEATURES)

        x = np.array([[float(feat_dict.get(n, 0.0)) for n in names]], dtype=np.float32)
        return self.predict(x)[0]

    # ------------------------------------------------------------------
    # Persistencia (legacy). En producción se recomienda guardar con train.py
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """
        Guarda el modelo entrenado a disco (pickle).

        Formato LEGACY:
        ---------------
        {
            "clf": self.clf,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "meta": self.meta (opcional)
        }

        Para modelos entrenados con `src/ml/train.py`, el guardado se hace
        allí usando joblib.dump(...) con el formato NUEVO.
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado, no se puede guardar.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "clf": self.clf,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "meta": self.meta,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"✅ Modelo (formato legacy) guardado en: {path}")

    @classmethod
    def load(cls, path: Path | str) -> "StanceClassifier":
        """
        Carga un modelo previamente guardado.

        Soporta ambos formatos:

        1) Formato NUEVO (generado por src/ml/train.py, vía joblib.dump):
           {
               "model": sklearn.Pipeline,
               "feature_cols": [...],
               "label_encoder": LabelEncoder,
               "meta": {...}
           }

        2) Formato LEGACY (guardado por StanceClassifier.save()):
           {
               "clf": RandomForestClassifier,
               "label_encoder": LabelEncoder,
               "feature_names": [...],
               "meta": {...}
           }
        """
        path = Path(path)
        data = joblib.load(path)  # soporta joblib y pickles simples

        inst = cls()

        if isinstance(data, dict):
            # Formato NUEVO (train.py)
            if "model" in data and "label_encoder" in data:
                inst.clf = data["model"]
                inst.label_encoder = data["label_encoder"]
                inst.feature_names = (
                    data.get("feature_cols")
                    or data.get("feature_names")
                    or list(BASE_FEATURES)
                )
                inst.meta = data.get("meta")
                inst.is_fitted = True
                print(f"✅ Modelo (formato NUEVO) cargado desde: {path}")
                return inst

            # Formato LEGACY (save())
            if "clf" in data and "label_encoder" in data:
                inst.clf = data["clf"]
                inst.label_encoder = data["label_encoder"]
                inst.feature_names = data.get("feature_names")
                inst.meta = data.get("meta")
                inst.is_fitted = True
                print(f"✅ Modelo (formato LEGACY) cargado desde: {path}")
                return inst

        raise RuntimeError(
            f"Formato de modelo no reconocido en {path}. "
            "¿Fue entrenado con src/ml/train.py o StanceClassifier.save()?"
        )

    # ------------------------------------------------------------------
    # Visualización
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[Path | str] = None,
    ) -> None:
        """
        Plotea y opcionalmente guarda la matriz de confusión.

        Importa matplotlib y seaborn solo aquí para que el resto del módulo
        pueda usarse sin estos paquetes (por ejemplo en producción).
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado. Usa load() o train().")

        import matplotlib.pyplot as plt
        import seaborn as sns  # type: ignore[import]

        y_pred = self.predict(X_test)
        cm = confusion_matrix(
            y_test, y_pred, labels=self.label_encoder.classes_
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix - Stance Classification")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Confusion matrix guardada en: {save_path}")
        plt.close()


# ----------------------------------------------------------------------
# Helper de entrenamiento desde CSV (modo rápido / legacy)
# ----------------------------------------------------------------------


def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Selecciona las columnas de features a partir de BASE_FEATURES,
    quedándose solo con las que realmente están en el CSV.

    Así si en el futuro agregas más columnas al CSV, este helper
    seguirá siendo robusto (y si falta alguna, lo avisa).
    """
    available = set(df.columns)
    base = list(BASE_FEATURES)
    feature_cols = [c for c in base if c in available]

    missing = [c for c in base if c not in available]
    if missing:
        print(
            f"⚠️  Advertencia: faltan columnas en CSV y se omiten: {missing}"
        )

    print(f"   Usando {len(feature_cols)} features: {feature_cols}")
    return feature_cols


def train_from_csv(
    csv_path: Path | str,
    model_output: Path | str,
    test_size: float = 0.2,
    cv_folds: int = 5,
    label_col: str = "stance_label",
) -> StanceClassifier:
    """
    Entrena el clasificador de posturas desde un CSV con features + labels
    (modo rápido / legacy).

    Para el pipeline principal del proyecto se recomienda usar `src/ml/train.py`.

    CSV esperado:
        - Columnas de features geométricas (coherentes con BASE_FEATURES).
        - Columna de etiqueta (por defecto: 'stance_label').

    Args:
        csv_path:
            Ruta al CSV con features de postura y etiquetas.
        model_output:
            Ruta donde guardar el modelo (pickle legacy).
        test_size:
            Fracción de datos reservados para test.
        cv_folds:
            Número de folds para cross-validation.
        label_col:
            Nombre de la columna de etiquetas en el CSV.

    Returns:
        Instancia de StanceClassifier ya entrenada.
    """
    csv_path = Path(csv_path)
    model_output = Path(model_output)

    print(f"📂 Cargando dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(
            f"Columna de etiqueta '{label_col}' no encontrada en {csv_path}"
        )

    # Selección de columnas de features en función de BASE_FEATURES
    feature_cols = _select_feature_cols(df)

    X = df[feature_cols].to_numpy(np.float32)
    y = df[label_col].to_numpy()

    print("\n📊 Dataset:")
    print(f"   Total muestras: {len(df)}")
    print(f"   Features:       {len(feature_cols)}")
    print(f"   Clases:         {np.unique(y)}")

    # Train/Test split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(
        f"\n📐 Train/Test split "
        f"({100 * (1 - test_size):.0f}/{100 * test_size:.0f}):"
    )
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test:  {len(X_test)} muestras")

    # Entrenar modelo (modo rápido)
    clf = StanceClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
    )
    metrics = clf.train(
        X_train,
        y_train,
        feature_names=feature_cols,
        cv_folds=cv_folds,
    )

    # Evaluar en test
    test_metrics = clf.evaluate(X_test, y_test)

    # Guardar modelo en formato legacy
    clf.meta = {
        "train_from_csv": True,
        "csv_path": str(csv_path),
        "feature_cols": feature_cols,
        "metrics_cv": metrics,
        "metrics_test": test_metrics,
    }
    clf.save(model_output)

    # Guardar confusion matrix
    cm_path = model_output.parent / "confusion_matrix_stances.png"
    clf.plot_confusion_matrix(X_test, y_test, save_path=cm_path)

    return clf


# ----------------------------------------------------------------------
# CLI sencillo para entrenamiento legacy desde CSV
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Entrenar clasificador de posturas (Random Forest, modo rápido / legacy).\n"
            "Para el pipeline principal se recomienda usar: python -m src.ml.train ..."
        )
    )
    ap.add_argument(
        "--csv",
        default="data/labels/stance_labels_auto.csv",
        help="CSV con features geométricas + stance_label",
    )
    ap.add_argument(
        "--output",
        default="data/models/stance_classifier.pkl",
        help="Archivo de salida (.pkl) para el modelo (formato legacy)",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fracción de datos destinada a test",
    )
    ap.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Número de folds para cross-validation",
    )
    args = ap.parse_args()

    clf = train_from_csv(
        csv_path=args.csv,
        model_output=args.output,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )

    print("\n🎉 Entrenamiento de clasificador de posturas (legacy) completado.")
