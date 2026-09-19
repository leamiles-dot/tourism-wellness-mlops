
import sys
import logging

from config import SPACE_REPO, HF_TOKEN
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    api = HfApi(token=HF_TOKEN)

    logger.info("Ensuring Space exists: %s", SPACE_REPO)
    api.create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=HF_TOKEN,
    )

    logger.info("Uploading deployment folder to %s", SPACE_REPO)
    api.upload_folder(
        folder_path="tourism_project/deployment",
        repo_id=SPACE_REPO,
        repo_type="space",
        token=HF_TOKEN,
    )

    logger.info("Deployment files pushed successfully to Space: %s", SPACE_REPO)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Hosting/deployment failed")
        sys.exit(1)
