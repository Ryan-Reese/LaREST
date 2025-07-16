import argparse
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from larest.helpers.constants import HARTTREE_TO_JMOL

# TODO: this needs to have a new file name/location


class CRESTParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__(description="Step 2 (CREST)")
        self._setup()

    def _setup(self) -> None:
        self.add_argument(
            "-o",
            "--output",
            type=str,
            default="./output",
            help="Output directory for Step 2",
        )
        self.add_argument(
            "-c",
            "--config",
            type=str,
            default="./config",
            help="Config directory for Step 2",
        )
        self.add_argument(
            "-v",
            "--verbose",
            help="increase output verbosity",
            action="store_true",
        )


class XTBParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__(description="Step 1 (xTB)")
        self._setup()

    def _setup(self) -> None:
        self.add_argument(
            "-o",
            "--output",
            type=str,
            default="./output",
            help="Output directory for Step 1",
        )
        self.add_argument(
            "-c",
            "--config",
            type=str,
            default="./config",
            help="Config directory for Step 1",
        )
        self.add_argument(
            "-v",
            "--verbose",
            help="increase output verbosity",
            action="store_true",
        )


def parse_monomer_smiles(args: argparse.Namespace, logger: logging.Logger) -> list[str]:
    input_file = os.path.join(args.config, "input.txt")
    logger.debug(f"Reading monomer smiles from {input_file}")

    try:
        with open(input_file) as fstream:
            monomer_smiles = fstream.read().splitlines()
    except Exception as err:
        logger.exception(err)
        logger.exception("Failed to read input monomer smiles")
        raise SystemExit(1) from err
    else:
        for i, smiles in enumerate(monomer_smiles):
            logger.debug(f"Read monomer {i}: {smiles}")
        return monomer_smiles


def parse_command_args(
    command_type: Literal[
        "xtb",
        "crest_confgen",
        "xtb_thermo",
        "confgen_xtb",
    ],
    config: dict[str, Any],
    logger: logging.Logger,
) -> list[str]:
    match command_type:
        case "xtb":
            params = config["step1"]["xtb"]
        case "crest_confgen":
            params = config["step2"]["crest"]["confgen"]
        case "xtb_thermo":
            params = config["step2"]["xtb"]["thermo"]
        case "confgen_xtb":
            params = config["step2"]["confgen"]["xtb"]

    args = []
    try:
        for k, v in params.items():
            if v is False:
                continue
            if v is True:
                args.append(f"--{k}")
            else:
                args.append(f"--{k}")
                args.append(str(v))

    except Exception as err:
        logger.exception(err)
        logger.warning(
            f"Failed to parse {command_type} arguments from dictionary {params}, using default arguments instead",
        )
        return []
    else:
        logger.debug(f"Returning {command_type} args: {args}")
        return args


def parse_xtb_output(
    xtb_output_file: str | Path,
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[float | None, float | None, float | None, float | None]:
    enthalpy, entropy, free_energy, total_energy = None, None, None, None

    logger.debug(f"Searching for results in file {xtb_output_file}")
    try:
        with open(xtb_output_file) as fstream:
            for i, line in enumerate(fstream):
                if "TOTAL ENERGY" in line:
                    try:
                        total_energy = float(line.split()[3]) * HARTTREE_TO_JMOL
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract total energy from line {i}: {line}",
                        )
                elif "TOTAL ENTHALPY" in line:
                    try:
                        enthalpy = float(line.split()[3]) * HARTTREE_TO_JMOL
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract total enthalpy from line {i}: {line}",
                        )
                elif "TOTAL FREE ENERGY" in line:
                    try:
                        free_energy = float(line.split()[4]) * HARTTREE_TO_JMOL
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract total free energy from line {i}: {line}",
                        )
    except Exception as err:
        logger.exception(err)
        logger.exception(f"Failed to parse xtb results from {xtb_output_file}")
        raise
    else:
        if enthalpy and free_energy:
            entropy = (enthalpy - free_energy) / config["step1"]["xtb"]["etemp"]
        if not (enthalpy and free_energy and total_energy):
            logger.warning(
                f"Failed to extract necessary data from {xtb_output_file}, missing data will be assigned None",
            )
        logger.debug(
            f"Found enthalpy: {enthalpy}, entropy: {entropy}, free_energy: {free_energy}, total energy: {total_energy}",
        )

    return enthalpy, entropy, free_energy, total_energy


def parse_most_stable_conformer(mol_dir: str | os.DirEntry) -> dict[str, float]:
    results_file = os.path.join(mol_dir, "post", "results.csv")

    results = pd.read_csv(
        results_file,
        header=0,
        index_col=False,
        dtype=np.float64,
    ).sort_values("free_energy", ascending=True)

    return results.iloc[0].to_dict()
