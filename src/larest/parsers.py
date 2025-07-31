import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from larest.constants import (
    CALMOL_TO_JMOL,
    CENSO_SECTIONS,
    CREST_ENTROPY_OUTPUT_PARAMS,
    HARTTREE_TO_JMOL,
    THERMODYNAMIC_PARAMS,
)


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
        logger.warning("Using default (no) arguments instead")
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
    xtb_output: dict[str, float | None] = dict.fromkeys(
        THERMODYNAMIC_PARAMS,
        None,
    )

    logger.debug(f"Searching for xTB results in file {xtb_output_file}")
    try:
        with open(xtb_output_file) as fstream:
            for i, line in enumerate(fstream):
                if "TOTAL ENTHALPY" in line:
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
            f"Found enthalpy: {xtb_output['H']}, entropy: {xtb_output['S']}, free energy {xtb_output['G']}",
        )

        return xtb_output


def parse_crest_entropy_output(
    crest_output_file: Path,
    logger: logging.Logger,
) -> dict[str, float | None]:
    crest_output: dict[str, float | None] = dict.fromkeys(
        CREST_ENTROPY_OUTPUT_PARAMS,
        None,
    )

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


def parse_censo_output(
    censo_output_file: Path,
    temperature: float,
    logger: logging.Logger,
) -> dict[str, dict[str, float | None]]:
    # WARN: cannot use fromkeys, otherwise they all point to the same mutable dict
    censo_output: dict[str, dict[str, float | None]] = {
        section: dict.fromkeys(THERMODYNAMIC_PARAMS, None) for section in CENSO_SECTIONS
    }

    logger.debug(f"Searching for CENSO results in file {censo_output_file}")
    try:
        with open(censo_output_file) as fstream:
            section_no: int = 0
            for i, line in enumerate(fstream):
                if f"part{section_no}" in line:
                    try:
                        censo_output[CENSO_SECTIONS[section_no]]["H"] = (
                            float(line.split()[1]) * HARTTREE_TO_JMOL
                        )
                        censo_output[CENSO_SECTIONS[section_no]]["G"] = (
                            float(line.split()[2]) * HARTTREE_TO_JMOL
                        )

                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract H and G from line {i}: {line}",
                        )
                    else:
                        section_no += 1
    except Exception as err:
        logger.exception(err)
        logger.exception(
            f"Failed to parse censo results from {censo_output_file}",
        )
        raise
    else:
        for params in censo_output.values():
            if params["H"] and params["G"]:
                params["S"] = (params["H"] - params["G"]) / temperature

            if not all(params.values()):
                logger.warning(
                    f"Failed to extract necessary data from {censo_output_file}",
                )
                logger.warning("Missing data will be assigned None")

            logger.debug(
                f"Found enthalpy: {params['H']}, free energy: {params['G']}, entropy {params['S']}",
            )

        return censo_output


def parse_best_censo_conformers(
    censo_output_file: Path,
    logger: logging.Logger,
) -> dict[str, str]:
    best_censo_conformers: dict[str, str] = dict.fromkeys(CENSO_SECTIONS, "CONF0")

    logger.debug(f"Searching for results in file {censo_output_file}")
    section_no: int = 0
    try:
        with open(censo_output_file) as fstream:
            for i, line in enumerate(fstream):
                if "Highest ranked conformer" in line:
                    try:
                        best_censo_conformers[CENSO_SECTIONS[section_no]] = (
                            line.split()[-1]
                        )
                    except Exception as err:
                        logger.exception(err)
                        logger.exception(
                            f"Failed to extract best conformer from line {i}: {line}",
                        )
                    else:
                        section_no += 1
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
            logger.warning("Missing sections will be assigned first conformer (CONF0)")

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
        f"Extracting best conformer ({best_conformer_id} .xyz from {censo_conformers_xyz_file}",
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
            f"Failed to extract best censo conformer xyz from {censo_conformers_xyz_file}",
        )
        raise
    else:
        logger.debug(
            f"Finished extracting best conformer xyz to {output_xyz_file}",
        )


def parse_best_rdkit_conformer(xtb_rdkit_results_file: Path) -> dict[str, float | None]:
    xtb_results_df: pd.DataFrame = pd.read_csv(
        xtb_rdkit_results_file,
        header=0,
        index_col=False,
        dtype=np.float64,
    ).sort_values("G", ascending=True)

    return xtb_results_df.iloc[0].to_dict()
