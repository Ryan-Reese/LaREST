import argparse
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from larest.parsers import parse_best_rdkit_conformer


def create_dir(dir_path: Path, logger: logging.Logger) -> None:
    # create specified dir
    logger.debug(f"Creating directory: {dir_path}")

    try:
        os.makedirs(dir_path, exist_ok=False)
    except FileExistsError:
        logger.warning(f"Directory {dir_path} already exists")
    else:
        logger.debug(f"Directory {dir_path} created")


def compile_monomer_results(
    monomer_smiles: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    data = dict(polymer=dict())
    step1_dir = os.path.join(args.output, "step1")
    monomer_dir = os.path.join(step1_dir, "monomer", slugify(monomer_smiles))
    data["monomer"] = parse_most_stable_conformer(monomer_dir)

    # get initiator results if ROR reaction
    if config["reaction"]["type"] == "ROR":
        initiator_dir = os.path.join(
            step1_dir,
            "initiator",
            slugify(config["reaction"]["initiator"]),
        )
        data["initiator"] = parse_most_stable_conformer(initiator_dir)

    parent_polymer_dir = os.path.join(step1_dir, "polymer")
    with os.scandir(parent_polymer_dir) as folder:
        polymer_dirs = [
            d
            for d in folder
            if (d.is_dir() and (slugify(monomer_smiles) in d.name.split("_")))
        ]

    for polymer_dir in polymer_dirs:
        polymer_length = polymer_dir.name.split("_")[1]
        data["polymer"][polymer_length] = parse_most_stable_conformer(polymer_dir)

    results = dict(
        polymer_length=[],
        monomer_enthalpy=[],
        initiator_enthalpy=[],
        polymer_enthalpy=[],
        monomer_entropy=[],
        initiator_entropy=[],
        polymer_entropy=[],
    )

    for polymer_length in data["polymer"].keys():
        results["polymer_length"].append(int(polymer_length))
        results["monomer_enthalpy"].append(data["monomer"]["enthalpy"])
        results["initiator_enthalpy"].append(
            data["initiator"]["enthalpy"]
            if (config["reaction"]["type"] == "ROR")
            else 0,
        )
        results["polymer_enthalpy"].append(data["polymer"][polymer_length]["enthalpy"])

        results["monomer_entropy"].append(data["monomer"]["entropy"])
        results["initiator_entropy"].append(
            data["initiator"]["entropy"]
            if (config["reaction"]["type"] == "ROR")
            else 0,
        )
        results["polymer_entropy"].append(data["polymer"][polymer_length]["entropy"])

    results = pd.DataFrame(results, index=None, dtype=np.float64).sort_values(
        "polymer_length",
        ascending=True,
    )
    results["delta_h"] = (
        results["polymer_enthalpy"]
        - (results["polymer_length"] * results["monomer_enthalpy"])
        - results["initiator_enthalpy"]
    ) / results["polymer_length"]

    results["delta_s"] = (
        results["polymer_entropy"]
        - (results["polymer_length"] * results["monomer_entropy"])
        - results["initiator_entropy"]
    ) / results["polymer_length"]

    # location to save results
    results_file = os.path.join(step1_dir, f"results_{slugify(monomer_smiles)}.csv")

    results.to_csv(results_file, header=True, index=False)


def get_most_stable_conformer_id(dir_name, args, logger):
    step1_post_dir = os.path.join(args.output, "step1", dir_name, "post")
    conformer_id = int(parse_most_stable_conformer(step1_post_dir)["conformer_id"])
    return conformer_id


# taken from django.utils
def slugify(smiles: str) -> str:
    smiles = (
        unicodedata.normalize("NFKD", smiles).encode("ascii", "ignore").decode("ascii")
    )
    smiles = re.sub(r"[^\w\s-]", "", smiles.lower())
    return re.sub(r"[-\s]+", "-", smiles).strip("-_")
