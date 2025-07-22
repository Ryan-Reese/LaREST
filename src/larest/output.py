import logging
import os
import re
import unicodedata
from pathlib import Path


def create_dir(dir_path: Path, logger: logging.Logger) -> None:
    # create specified dir
    logger.debug(f"Creating directory: {dir_path}")

    try:
        os.makedirs(dir_path, exist_ok=False)
    except FileExistsError:
        logger.warning(f"Directory {dir_path} already exists")
    else:
        logger.debug(f"Directory {dir_path} created")


# taken from django.utils
def slugify(smiles: str) -> str:
    smiles = (
        unicodedata.normalize("NFKD", smiles).encode("ascii", "ignore").decode("ascii")
    )
    smiles = re.sub(r"[^\w\s-]", "", smiles.lower())
    return re.sub(r"[-\s]+", "-", smiles).strip("-_")
