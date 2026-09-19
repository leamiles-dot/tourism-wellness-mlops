
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATASET_REPO, HF_TOKEN, RANDOM_STATE, TARGET_COL, TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns, fix known data-entry issues, and cap extreme outliers."""
    df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

    # Fix a data-entry typo found during EDA ("Fe Male" -> "Female")
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Merge "Unmarried" into "Single" -- these appear to be duplicate labels for the same status
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Cap extreme outliers at the 99th percentile
    for col in ["MonthlyIncome", "DurationOfPitch", "NumberOfTrips"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    return df


def main():
    logger.info("Downloading raw dataset from %s", DATASET_REPO)
    raw_path = hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset", filename="tourism.csv", token=HF_TOKEN
    )
    df = pd.read_csv(raw_path)
    logger.info("Loaded raw dataset with shape %s", df.shape)

    df = clean(df)
    logger.info("Cleaned dataset shape: %s", df.shape)

    # 1st split: carve off the TEST set. It will not be touched again until final evaluation in train.py
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )

    # 2nd split: divide the remainder into TRAIN and VALIDATION.
    # Recalculate the proportion so validation still ends up as VALIDATION_SIZE of the ORIGINAL data,
    # not of the already-reduced train_val_df.
    relative_val_size = VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)
    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=RANDOM_STATE,
        stratify=train_val_df[TARGET_COL],
    )

    logger.info(
        "Split sizes -> train: %d, validation: %d, test: %d",
        len(train_df), len(validation_df), len(test_df),
    )

    os.makedirs("tourism_project/data", exist_ok=True)
    train_df.to_csv("tourism_project/data/train.csv", index=False)
    validation_df.to_csv("tourism_project/data/validation.csv", index=False)
    test_df.to_csv("tourism_project/data/test.csv", index=False)

    api = HfApi(token=HF_TOKEN)
    for fname in ["train.csv", "validation.csv", "test.csv"]:
        logger.info("Uploading %s to %s", fname, DATASET_REPO)
        api.upload_file(
            path_or_fileobj=f"tourism_project/data/{fname}",
            path_in_repo=fname,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )

    logger.info("Data preparation complete. train/validation/test pushed to %s", DATASET_REPO)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Data preparation failed")
        sys.exit(1)
