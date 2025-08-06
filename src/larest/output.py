import os
import re
import shutil
import unicodedata
from logging import Logger
from pathlib import Path


def create_dir(dir_path: Path, logger: Logger) -> None:
    # create specified dir
    logger.debug(f"Creating directory: {dir_path}")

    try:
        os.makedirs(dir_path, exist_ok=False)
    except FileExistsError:
        logger.warning(f"Directory {dir_path} already exists")
    else:
        logger.debug(f"Directory {dir_path} created")


def remove_dir(dir_path: Path, logger: Logger) -> None:
    # remove specified dir
    logger.debug(f"Removing directory: {dir_path}")

    try:
        shutil.rmtree(dir_path, ignore_errors=False)
    except Exception:
        logger.warning(f"Failed to remove directory {dir_path}")
    else:
        logger.debug(f"Directory {dir_path} removed")


# modified from django.utils
def slugify(smiles: str) -> str:
    smiles = (
        unicodedata.normalize("NFKD", smiles).encode("ascii", "ignore").decode("ascii")
    )
    smiles = re.sub(r"[\(\)]", "-", smiles)
    smiles = re.sub(r"[^\w\s@-]", "", smiles)
    return re.sub(r"[-\s]+", "-", smiles).strip("-_")
