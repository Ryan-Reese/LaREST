import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from larest.constants import (
    CALMOL_TO_JMOL,
    CENSO_SECTIONS,
    CREST_OUTPUT_PARAMS,
    HARTTREE_TO_JMOL,
    XTB_OUTPUT_PARAMS,
)

# TODO: this needs to have a new file name/location


class LarestArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__(description="LaREST")
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


# def parse_monomer_smiles(args: argparse.Namespace, logger: logging.Logger) -> list[str]:
#     input_file = os.path.join(args.config, "input.txt")
#     logger.debug(f"Reading monomer smiles from {input_file}")
#
#     try:
#         with open(input_file) as fstream:
#             monomer_smiles = fstream.read().splitlines()
#     except Exception as err:
#         logger.exception(err)
#         logger.exception("Failed to read input monomer smiles")
#         raise SystemExit(1) from err
#     else:
#         for i, smiles in enumerate(monomer_smiles):
#             logger.debug(f"Read monomer {i}: {smiles}")
# return monomer_smiles


def parse_command_args(
    sub_config: list[str],
    config: dict[str, Any],
    logger: logging.Logger,
) -> list[str]:
    try:
        for config_key in sub_config:
            config = config[config_key]
    except Exception as err:
        logger.exception(err)
        logger.warning(f"Failed to find sub-config {sub_config} in config {config}")
        logger.warning("Using default arguments instead")
        return []

    args = []
    for k, v in config.items():
        if v is False:
            continue
        if v is True:
            args.append(f"--{k}")
        else:
            args.append(f"--{k}")
            args.append(str(v))
    logger.debug(f"Returning {sub_config} args: {args}")
    return args


def parse_xtb_output(
    xtb_output_file: Path,
    temperature: float,
    logger: logging.Logger,
) -> dict[str, float | None]:
    xtb_output: dict[str, float | None] = dict.fromkeys(XTB_OUTPUT_PARAMS, None)

    logger.debug(f"Searching for results in file {xtb_output_file}")
    try:
        with open(xtb_output_file) as fstream:
            for i, line in enumerate(fstream):
                if "TOTAL ENERGY" in line:
                    try:
                        xtb_output["E"] = float(line.split()[3]) * HARTTREE_TO_JMOL
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract total energy from line {i}: {line}",
                        )
                elif "TOTAL ENTHALPY" in line:
                    try:
                        xtb_output["H"] = float(line.split()[3]) * HARTTREE_TO_JMOL
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract total enthalpy from line {i}: {line}",
                        )
                elif "TOTAL FREE ENERGY" in line:
                    try:
                        xtb_output["G"] = float(line.split()[4]) * HARTTREE_TO_JMOL
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
        if xtb_output["H"] and xtb_output["G"]:
            xtb_output["S"] = (xtb_output["H"] - xtb_output["G"]) / temperature
        if not all(xtb_output.values()):
            logger.warning(f"Failed to extract necessary data from {xtb_output_file}")
            logger.warning("Missing data will be assigned None")
        logger.debug(
            f"Found enthalpy: {xtb_output['H']}, entropy: {xtb_output['S']}",
        )
        logger.debug(
            f"Free energy {xtb_output['G']}, total energy {xtb_output['E']}",
        )

        return xtb_output


def parse_crest_entropy_output(
    crest_output_file: Path,
    logger: logging.Logger,
) -> dict[str, float | None]:
    crest_output: dict[str, float | None] = dict.fromkeys(CREST_OUTPUT_PARAMS, None)

    logger.debug(f"Searching for results in file {crest_output_file}")
    try:
        with open(crest_output_file) as fstream:
            for i, line in enumerate(fstream):
                if "Sconf" in line:
                    try:
                        crest_output["S_conf"] = (
                            float(line.split()[-1]) * CALMOL_TO_JMOL
                        )
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract S_conf from line {i}: {line}",
                        )
                elif ("+" in line) and ("δSrrho" in line) and (len(line.split()) == 4):
                    try:
                        crest_output["S_rrho"] = (
                            float(line.split()[-1]) * CALMOL_TO_JMOL
                        )
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract S_rrho from line {i}: {line}",
                        )
                elif ("S(total)" in line) and ("cal" in line):
                    try:
                        crest_output["S_total"] = (
                            float(line.split()[3]) * CALMOL_TO_JMOL
                        )
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract S_total from line {i}: {line}",
                        )
    except Exception as err:
        logger.exception(err)
        logger.exception(
            f"Failed to parse crest entropy results from {crest_output_file}",
        )
        raise
    else:
        if not all(crest_output.values()):
            logger.warning(f"Failed to extract necessary data from {crest_output_file}")
            logger.warning("Missing data will be assigned None")
        logger.debug(
            f"Found S_conf: {crest_output['S_conf']}, S_rrho: {crest_output['S_rrho']}",
        )
        logger.debug(
            f"S_total {crest_output['S_total']}",
        )

        return crest_output


def parse_best_censo_conformers(
    censo_output_file: Path,
    logger: logging.Logger,
) -> dict[str, str]:
    best_censo_conformers: dict[str, str] = dict.fromkeys(CENSO_SECTIONS, "CONF0")

    logger.debug(f"Searching for results in file {censo_output_file}")
    censo_section_id: int = 0
    try:
        with open(censo_output_file) as fstream:
            for i, line in enumerate(fstream):
                if "Highest ranked conformer" in line:
                    try:
                        best_censo_conformers[CENSO_SECTIONS[censo_section_id]] = (
                            line.split()[-1]
                        )
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract best conformer from line {i}: {line}",
                        )
                    else:
                        censo_section_id += 1
    except Exception as err:
        logger.exception(err)
        logger.exception(
            f"Failed to determine best censo conformers from {censo_output_file}",
        )
        raise
    else:
        if not all(best_censo_conformers.values()):
            logger.warning(
                f"Failed to extract best conformers from {censo_output_file}",
            )
            logger.warning("Missing data will be assigned CONF0")

        for section in CENSO_SECTIONS:
            logger.debug(
                f"Best conformer in {section}: {best_censo_conformers[section]}",
            )

        return best_censo_conformers


def extract_best_conformer_xyz(
    censo_conformers_xyz_file: Path,
    best_conformer_id: str,
    output_xyz_file: Path,
    logger: logging.Logger,
) -> None:
    logger.debug(
        f"Extracting best conformer ({best_conformer_id} from {censo_conformers_xyz_file}",
    )
    try:
        with open(censo_conformers_xyz_file) as fin:
            conformers_xyz: list[str] = fin.readlines()
            n_atoms: int = int(conformers_xyz[0])
            for i, line in enumerate(conformers_xyz):
                if best_conformer_id in line.split():
                    try:
                        with open(output_xyz_file, "w") as fout:
                            fout.writelines(conformers_xyz[i - 1 : i + n_atoms + 2])
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to write best conformer to {output_xyz_file}",
                        )
                    break
    except Exception as err:
        logger.exception(err)
        logger.exception(
            f"Failed to extract best censo conformer from {censo_conformers_xyz_file}",
        )
        raise
    else:
        logger.debug(
            f"Finished extracting best conformer to {output_xyz_file}",
        )


def parse_best_rdkit_conformer(xtb_rdkit_dir: Path) -> dict[str, float]:
    xtb_results_file: Path = xtb_rdkit_dir / "results.csv"

    xtb_results_df: pd.DataFrame = pd.read_csv(
        xtb_results_file,
        header=0,
        index_col=False,
        dtype=np.float64,
    ).sort_values("G", ascending=True)

    return xtb_results_df.iloc[0].to_dict()
