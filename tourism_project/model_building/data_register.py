
import os
import sys
import logging

# config.py lives one directory up (tourism_project/), so add that to the import path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATASET_REPO, HF_TOKEN

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    api = HfApi(token=HF_TOKEN)

    # Create the dataset repo if it does not already exist (safe to re-run)
    logger.info("Ensuring dataset repo exists: %s", DATASET_REPO)
    api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

    raw_path = "tourism_project/data/tourism.csv"
    logger.info("Uploading raw dataset from %s to %s", raw_path, DATASET_REPO)
    api.upload_file(
        path_or_fileobj=raw_path,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    logger.info("Raw dataset registered successfully at %s", DATASET_REPO)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Data registration failed")
        sys.exit(1)
