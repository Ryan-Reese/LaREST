import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from larest.helpers.parsers import parse_most_stable_conformer


def create_dir(dir_path: str | Path, logger: logging.Logger) -> None:
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
    monomer_dir = os.path.join(step1_dir, "monomer", monomer_smiles)
    data["monomer"] = parse_most_stable_conformer(monomer_dir)

    # get initiator results if ROR reaction
    if config["reaction"]["type"] == "ROR":
        initiator_dir = os.path.join(
            step1_dir, "initiator", config["initiator"]["smiles"]
        )
        data["initiator"] = parse_most_stable_conformer(initiator_dir)

    parent_polymer_dir = os.path.join(step1_dir, "polymer")
    with os.scandir(parent_polymer_dir) as folder:
        polymer_dirs = [
            d for d in folder if (d.is_dir() and (monomer_smiles in d.name.split("_")))
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
            else 0
        )
        results["polymer_enthalpy"].append(data["polymer"][polymer_length]["enthalpy"])

        results["monomer_entropy"].append(data["monomer"]["entropy"])
        results["initiator_entropy"].append(
            data["initiator"]["entropy"] if (config["reaction"]["type"] == "ROR") else 0
        )
        results["polymer_entropy"].append(data["polymer"][polymer_length]["entropy"])

    results = pd.DataFrame(results, index=None, dtype=np.float64).sort_values(
        "polymer_length", ascending=True
    )
    results["delta_h2"] = (
        results["polymer_enthalpy"]
        - (results["polymer_length"] * results["monomer_enthalpy"])
        - results["initiator_enthalpy"]
    )

    results["delta_s"] = (
        results["polymer_entropy"]
        - (results["polymer_length"] * results["monomer_entropy"])
        - results["initiator_entropy"]
    )

    # location to save results
    results_file = os.path.join(step1_dir, f"results_{monomer_smiles}.csv")

    results.to_csv(results_file, header=True, index=False)


def get_most_stable_conformer_id(dir_name, args, logger):
    step1_post_dir = os.path.join(args.output, "step1", dir_name, "post")
    conformer_id = int(parse_most_stable_conformer(step1_post_dir)["conformer_id"])
    return conformer_id


def copy_most_stable_conformer(dir_name, args, logger):
    step1_mol_dir = os.path.join(args.output, "step1", dir_name)
    step2_pre_dir = os.path.join(args.output, "step2", dir_name, "pre")
    conformer_id = int(parse_most_stable_conformer(step1_mol_dir)["conformer_id"])
    conformer_xyz_file = os.path.join(
        step1_mol_dir,
        "post",
        f"conformer_{conformer_id}",
        f"conformer_{conformer_id}.xtbopt.xyz",
    )
    return shutil.copy2(conformer_xyz_file, step2_pre_dir)
