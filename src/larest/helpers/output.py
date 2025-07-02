import logging
import os
from pathlib import Path


def create_dir(dir_path: str | Path, logger: logging.Logger) -> None:
    # create specified dir
    logger.debug(f"Creating directory: {dir_path}")

    try:
        os.makedirs(dir_path, exist_ok=False)
    except FileExistsError:
        logger.warning(f"Directory {dir_path} already exists")
    else:
        logger.debug(f"Directory {dir_path} created")
