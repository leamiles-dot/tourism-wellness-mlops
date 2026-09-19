
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    DATASET_REPO, MODEL_REPO, HF_TOKEN, RANDOM_STATE, TARGET_COL, MODEL_NAMES, PARAM_GRIDS,
)

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_split(filename: str) -> pd.DataFrame:
    path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=filename, token=HF_TOKEN)
    return pd.read_csv(path)


def evaluate(pipeline, X, y) -> dict:
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probs),
    }


def build_pipeline(categorical_cols, model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
        remainder="passthrough",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_model(name: str, scale_pos_weight: float):
    """Return a FRESH, unfitted instance of the requested algorithm.

    Class imbalance is handled via class_weight="balanced" for algorithms that support it
    natively; XGBoost uses its own scale_pos_weight parameter instead, since it has no
    class_weight argument. Gradient Boosting has no equivalent in scikit-learn at all, which
    is a known limitation of that algorithm here -- its ranking may be penalised for it.
    """
    if name == "Decision Tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    if name == "Bagging":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        )
    if name == "Random Forest":
        return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    if name == "AdaBoost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    if name == "XGBoost":
        return XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric="logloss")
    raise ValueError(f"Unknown model name: {name}")


def get_feature_importances(fitted_model, feature_names):
    """Return a Series of feature importances, with a fallback for models (like Bagging)
    that don't expose feature_importances_ directly but whose base estimators do."""
    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
    elif hasattr(fitted_model, "estimators_"):
        values = np.mean([est.feature_importances_ for est in fitted_model.estimators_], axis=0)
    else:
        return None
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def main():
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)

    # 1. Load train, validation, and test SEPARATELY -- each has one job and one job only
    logger.info("Loading train / validation / test splits from %s", DATASET_REPO)
    train_df = load_split("train.csv")
    validation_df = load_split("validation.csv")
    test_df = load_split("test.csv")
    logger.info(
        "Loaded shapes -> train: %s, validation: %s, test: %s",
        train_df.shape, validation_df.shape, test_df.shape,
    )

    X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_val, y_val = validation_df.drop(columns=[TARGET_COL]), validation_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    # -1 = always used for fitting (train rows); 0 = held out for scoring (validation rows)
    test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
    predefined_split = PredefinedSplit(test_fold)

    mlflow.set_experiment("tourism-wellness-prediction")

    with mlflow.start_run():
        baseline_results = []
        tuned_results = []
        fitted_baseline = {}
        fitted_tuned = {}

        # ------------------------------------------------------------------
        # 2. For EACH of the six algorithms: train a baseline, then tune it
        # ------------------------------------------------------------------
        for name in MODEL_NAMES:
            safe_name = name.replace(" ", "_")
            logger.info("=== %s ===", name)

            # --- baseline: fixed, untuned parameters, fit on TRAIN only ---
            baseline_pipeline = build_pipeline(categorical_cols, build_model(name, scale_pos_weight))
            baseline_pipeline.fit(X_train, y_train)
            baseline_val_f1 = f1_score(y_val, baseline_pipeline.predict(X_val))
            baseline_test_metrics = evaluate(baseline_pipeline, X_test, y_test)

            mlflow.log_metric(f"baseline_{safe_name}_validation_f1", baseline_val_f1)
            mlflow.log_metric(f"baseline_{safe_name}_test_f1", baseline_test_metrics["f1"])

            baseline_results.append({
                "model": name, "validation_f1": baseline_val_f1,
                **{f"test_{k}": v for k, v in baseline_test_metrics.items()},
            })
            fitted_baseline[name] = baseline_pipeline
            logger.info("  baseline  -> validation F1: %.4f | test F1: %.4f", baseline_val_f1, baseline_test_metrics["f1"])

            # --- tuned: GridSearchCV against the VALIDATION set ---
            tuning_pipeline = build_pipeline(categorical_cols, build_model(name, scale_pos_weight))
            grid_search = GridSearchCV(
                tuning_pipeline, PARAM_GRIDS[name], cv=predefined_split, scoring="f1", n_jobs=-1
            )
            grid_search.fit(X_train_val, y_train_val)

            tuned_val_f1 = grid_search.best_score_
            tuned_test_metrics = evaluate(grid_search.best_estimator_, X_test, y_test)

            mlflow.log_metric(f"tuned_{safe_name}_validation_f1", tuned_val_f1)
            mlflow.log_metric(f"tuned_{safe_name}_test_f1", tuned_test_metrics["f1"])
            mlflow.log_param(f"tuned_{safe_name}_best_params", str(grid_search.best_params_))

            tuned_results.append({
                "model": name, "validation_f1": tuned_val_f1,
                "best_params": str(grid_search.best_params_),
                **{f"test_{k}": v for k, v in tuned_test_metrics.items()},
            })
            fitted_tuned[name] = grid_search.best_estimator_
            logger.info(
                "  tuned     -> validation F1: %.4f | test F1: %.4f | params: %s",
                tuned_val_f1, tuned_test_metrics["f1"], grid_search.best_params_,
            )

        # ------------------------------------------------------------------
        # 3. Rank each stage by VALIDATION F1 (never by test F1)
        # ------------------------------------------------------------------
        baseline_ranking = pd.DataFrame(baseline_results).sort_values(
            "validation_f1", ascending=False
        ).reset_index(drop=True)
        baseline_ranking.insert(0, "rank", baseline_ranking.index + 1)

        tuned_ranking = pd.DataFrame(tuned_results).sort_values(
            "validation_f1", ascending=False
        ).reset_index(drop=True)
        tuned_ranking.insert(0, "rank", tuned_ranking.index + 1)

        logger.info("Baseline ranking (by validation F1):\n%s",
                    baseline_ranking[["rank", "model", "validation_f1", "test_f1"]].to_string(index=False))
        logger.info("Tuned ranking (by validation F1):\n%s",
                    tuned_ranking[["rank", "model", "validation_f1", "test_f1"]].to_string(index=False))

        # 4. Combine both stages and pick the SINGLE overall best model, by validation F1
        overall_ranking = pd.concat([
            baseline_ranking.assign(stage="baseline"),
            tuned_ranking.assign(stage="tuned"),
        ], ignore_index=True).sort_values("validation_f1", ascending=False).reset_index(drop=True)
        overall_ranking.insert(0, "overall_rank", overall_ranking.index + 1)

        logger.info("Overall ranking (baseline + tuned combined, by validation F1):\n%s",
                    overall_ranking[["overall_rank", "model", "stage", "validation_f1", "test_f1"]].to_string(index=False))

        best_row = overall_ranking.iloc[0]
        best_model_name = best_row["model"]
        best_stage = best_row["stage"]
        best_pipeline = fitted_tuned[best_model_name] if best_stage == "tuned" else fitted_baseline[best_model_name]

        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("selected_stage", best_stage)
        logger.info("SELECTED MODEL: %s (%s) -- validation F1 %.4f", best_model_name, best_stage, best_row["validation_f1"])

        # 5. Final, ONE-TIME test evaluation -- of the SELECTED model only
        final_test_metrics = evaluate(best_pipeline, X_test, y_test)
        mlflow.log_metrics({f"selected_test_{k}": v for k, v in final_test_metrics.items()})
        logger.info("Final test metrics for selected model: %s", final_test_metrics)

        os.makedirs("tourism_project/model_building/artifacts", exist_ok=True)
        baseline_ranking.to_csv("tourism_project/model_building/artifacts/baseline_ranking.csv", index=False)
        tuned_ranking.to_csv("tourism_project/model_building/artifacts/tuned_ranking.csv", index=False)
        overall_ranking.to_csv("tourism_project/model_building/artifacts/overall_ranking.csv", index=False)
        mlflow.log_artifact("tourism_project/model_building/artifacts/baseline_ranking.csv")
        mlflow.log_artifact("tourism_project/model_building/artifacts/tuned_ranking.csv")
        mlflow.log_artifact("tourism_project/model_building/artifacts/overall_ranking.csv")

        # 6. Feature importance for the SELECTED model (with a fallback for models like
        #    Bagging that don't expose feature_importances_ directly)
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        importance_series = get_feature_importances(best_pipeline.named_steps["model"], feature_names)

        importance_csv_path = "tourism_project/model_building/artifacts/feature_importance.csv"
        if importance_series is not None:
            top_importances = importance_series.head(15)
            top_importances.rename("importance").to_csv(importance_csv_path, header=True)

            plt.figure(figsize=(8, 6))
            top_importances.sort_values().plot(kind="barh")
            plt.title(f"Top 15 Feature Importances ({best_model_name})")
            plt.xlabel("Importance")
            plt.tight_layout()
            importance_png_path = "tourism_project/model_building/artifacts/feature_importance.png"
            plt.savefig(importance_png_path)
            plt.close()
            mlflow.log_artifact(importance_png_path)
            mlflow.log_artifact(importance_csv_path)
            logger.info("Top 5 most important features:\n%s", top_importances.head(5))
        else:
            logger.warning("Selected model type exposes no feature importances; skipping importance chart.")
            pd.Series(dtype=float, name="importance").to_csv(importance_csv_path, header=True)

        # skops_trusted_types is needed because MLflow's model-logging security scanner
        # does not trust XGBoost's internal types by default -- without this, logging would
        # crash whenever XGBoost happens to be the selected model
        mlflow.sklearn.log_model(
            best_pipeline, "model",
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
        )

    # 7. Save the SELECTED pipeline locally, then register it (plus ranking tables) on Hugging Face
    model_path = "tourism_project/model_building/artifacts/model.joblib"
    joblib.dump(best_pipeline, model_path)

    api.upload_file(path_or_fileobj=model_path, path_in_repo="model.joblib",
                     repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN)
    api.upload_file(path_or_fileobj=importance_csv_path, path_in_repo="feature_importance.csv",
                     repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj="tourism_project/model_building/artifacts/overall_ranking.csv",
        path_in_repo="model_comparison_ranking.csv",
        repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN,
    )

    logger.info("Selected model (%s, %s) and comparison artifacts pushed to %s",
                best_model_name, best_stage, MODEL_REPO)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Model training failed")
        sys.exit(1)
