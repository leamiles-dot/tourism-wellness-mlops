
import os
import sys

from huggingface_hub import whoami

# Populated by the getpass() cell locally, or by the GitHub Actions secret in CI
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    sys.exit(
        "HF_TOKEN is not set. Run the getpass() cell (locally) or check the GitHub Actions "
        "secret (in CI) before running any script that imports config.py."
    )

# --- Hugging Face repos --------------------------------------------------
# Username is derived from the token itself, not hardcoded, so it can never drift out of
# sync with whichever Hugging Face account this token actually belongs to.
try:
    HF_USERNAME = whoami(token=HF_TOKEN)["name"]
except Exception as e:
    sys.exit(f"Could not verify Hugging Face identity from HF_TOKEN: {e}")

DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"
MODEL_REPO = f"{HF_USERNAME}/tourism-wellness-model"
SPACE_REPO = f"{HF_USERNAME}/tourism-wellness-app"

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE = 42
TARGET_COL = "ProdTaken"

# --- Train / validation / test split proportions ---------------------------
# 70% train, 15% validation, 15% test -- validation is used ONLY for
# hyperparameter selection; test is touched exactly once, at the very end.
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# --- Algorithms being compared -----------------------------------------------
# The rubric explicitly allows any of these six algorithm families
MODEL_NAMES = [
    "Decision Tree",
    "Bagging",
    "Random Forest",
    "AdaBoost",
    "Gradient Boosting",
    "XGBoost",
]

# --- Hyperparameter grid to tune over, PER algorithm ------------------------
PARAM_GRIDS = {
    "Decision Tree": {
        "model__max_depth": [5, 10],
        "model__min_samples_split": [2, 5],
    },
    "Bagging": {
        "model__n_estimators": [10, 30],
        "model__max_samples": [0.7, 1.0],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20],
    },
    "AdaBoost": {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.5, 1.0],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
    },
}
