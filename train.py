"""Train, evaluate and export the churn model.

Usage::

    python train.py                      # defaults: data/telco_churn.csv -> models/
    python train.py --data my.csv --out models

Steps
-----
1. Load and clean the Telco dataset (see ``churn/features.py``).
2. Stratified 80/20 train/holdout split.
3. For each candidate model, 5-fold cross-validated ROC-AUC on the training split.
4. Pick the best candidate by CV AUC. Choose the decision threshold that maximises
   F1 on *out-of-fold* predictions (never on the holdout), then report holdout metrics.
5. Permutation feature importance and business-impact numbers on the holdout.
6. Save ``models/churn_pipeline.joblib`` and ``models/metrics.json``.

The exported pipeline includes preprocessing, so the API feeds it raw customer
rows and never has to replicate any feature engineering.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.features import (
    ALL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORY_VALUES,
    DATA_PATH,
    NUMERIC_FEATURES,
    TARGET,
    clean,
    load_raw,
)

RANDOM_STATE = 42
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "models"


def build_preprocessor() -> ColumnTransformer:
    """Scale numerics, pass the 0/1 flag through, one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[CATEGORY_VALUES[c] for c in CATEGORICAL_FEATURES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def candidate_models() -> dict[str, object]:
    """The models we compare. Keys become the ``best_model`` name in metrics.json."""
    return {
        "logistic_regression": LogisticRegression(max_iter=5000, C=0.5),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=4,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(model: object) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def classification_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }


def best_f1_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Threshold that maximises F1. Called on out-of-fold predictions only."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one more point than thresholds; drop it.
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return round(float(thresholds[int(np.argmax(f1))]), 4)


def business_impact(
    holdout: pd.DataFrame, y_true: pd.Series, proba: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Translate the confusion matrix into monthly-revenue terms.

    "Revenue capture rate" answers the question a retention team actually asks:
    of the monthly revenue that walked out the door, what share would this model
    have flagged in time to act on?
    """
    monthly = holdout["MonthlyCharges"].to_numpy(dtype=float)
    flagged = proba >= threshold
    churned = y_true.to_numpy().astype(bool)

    revenue_churning = float(monthly[churned].sum())
    revenue_caught = float(monthly[churned & flagged].sum())
    revenue_flagged = float(monthly[flagged].sum())
    return {
        "holdout_customers": int(len(holdout)),
        "holdout_churners": int(churned.sum()),
        "customers_flagged": int(flagged.sum()),
        "flag_rate": round(float(flagged.mean()), 4),
        "monthly_revenue_of_churners": round(revenue_churning, 2),
        "monthly_revenue_of_churners_flagged": round(revenue_caught, 2),
        "revenue_capture_rate": round(revenue_caught / revenue_churning, 4) if revenue_churning else 0.0,
        "monthly_revenue_of_all_flagged": round(revenue_flagged, 2),
        "flag_revenue_precision": round(revenue_caught / revenue_flagged, 4) if revenue_flagged else 0.0,
    }


def train(data_path: Path, out_dir: Path) -> dict:
    started = time.perf_counter()
    df = clean(load_raw(data_path))
    X, y = df[ALL_FEATURES], df[TARGET]
    print(f"Loaded {len(df):,} customers, churn rate {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in candidate_models().items():
        pipe = make_pipeline(model)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        holdout = classification_metrics(y_test, pipe.predict_proba(X_test)[:, 1], 0.5)
        candidates[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "holdout_at_0.5": holdout,
        }
        fitted[name] = pipe
        print(
            f"  {name:<24} CV AUC {cv_auc.mean():.4f} ± {cv_auc.std():.4f} | "
            f"holdout AUC {holdout['roc_auc']:.4f} acc {holdout['accuracy']:.4f}"
        )

    best_name = max(candidates, key=lambda n: candidates[n]["cv_roc_auc_mean"])
    best = fitted[best_name]
    print(f"Best model by CV AUC: {best_name}")

    # Tune the operating threshold on out-of-fold predictions from the training
    # split so the holdout numbers below stay an honest estimate.
    oof = cross_val_predict(
        make_pipeline(candidate_models()[best_name]), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    threshold = best_f1_threshold(y_train, oof)

    proba_test = best.predict_proba(X_test)[:, 1]
    tuned = classification_metrics(y_test, proba_test, threshold)
    cm = confusion_matrix(y_test, (proba_test >= threshold).astype(int))
    print(
        f"Threshold {threshold:.3f}: acc {tuned['accuracy']:.4f} precision {tuned['precision']:.4f} "
        f"recall {tuned['recall']:.4f} f1 {tuned['f1']:.4f} AUC {tuned['roc_auc']:.4f}"
    )

    perm = permutation_importance(
        best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE
    )
    importance = sorted(
        (
            {"feature": f, "importance": round(float(m), 5), "std": round(float(s), 5)}
            for f, m, s in zip(ALL_FEATURES, perm.importances_mean, perm.importances_std)
        ),
        key=lambda d: -d["importance"],
    )

    impact = business_impact(X_test, y_test, proba_test, threshold)
    print(
        f"Business impact: flags {impact['flag_rate']:.1%} of customers and captures "
        f"{impact['revenue_capture_rate']:.1%} of churning monthly revenue"
    )

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(time.perf_counter() - started, 1),
        "dataset": {
            "path": str(data_path),
            "rows": int(len(df)),
            "churn_rate": round(float(y.mean()), 4),
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_test)),
        },
        "best_model": best_name,
        "threshold": threshold,
        "holdout": tuned,
        "confusion_matrix": {
            "labels": ["stayed", "churned"],
            "matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "candidates": candidates,
        "feature_importance": importance,
        "business_impact": impact,
        "features": ALL_FEATURES,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out_dir / "churn_pipeline.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved pipeline + metrics to {out_dir} in {metrics['training_seconds']}s")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the Telco churn CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Directory for model artifacts")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
