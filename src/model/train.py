from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "churn"
ID_COLUMNS = ["customer_id", "company_name"]
NUMERIC_FEATURES = [
    "contract_value",
    "monthly_recurring_revenue",
    "tenure_months",
    "product_usage_score",
    "support_ticket_count",
    "unresolved_ticket_count",
    "payment_delay_days",
    "nps_score",
    "last_login_days",
    "account_health_score",
]
CATEGORICAL_FEATURES = ["industry", "contract_type", "onboarding_completed"]


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_churn_model(
    customers: pd.DataFrame,
    output_dir: str | Path = "data/processed",
) -> dict[str, object]:
    features = customers.drop(columns=[TARGET, *ID_COLUMNS])
    target = customers[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_rows": int(len(y_test)),
        "positive_rate": round(float(target.mean()), 4),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "churn_model.joblib"
    metrics_path = output_path / "model_metrics.json"
    importance_path = output_path / "feature_importance.csv"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    feature_importance(pipeline).to_csv(importance_path, index=False)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "importance_path": str(importance_path),
        "metrics": metrics,
    }


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    clean_names = [name.replace("numeric__", "").replace("categorical__", "") for name in feature_names]
    return (
        pd.DataFrame({"feature": clean_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

