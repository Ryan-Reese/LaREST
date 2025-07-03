import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

from censo.emsembleopt import Optimization, Prescreening, Refinement, Screening
from censo.ensembledata import EnsembleData
from censo.params import Config
from censo.properties import NMR

from larest.helpers.output import copy_most_stable_conformer, create_dir
from larest.helpers.parsers import (
    CRESTParser,
    parse_command_args,
    parse_monomer_smiles,
)
from larest.helpers.setup import get_config, get_logger


def run_censo_conformers(
    smiles: str,
    mol_type: Literal["monomer", "initiator", "polymer"],
    dir_name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
):
    # Create output dirs
    mol_dir = os.path.join(args.output, "step2", dir_name)
    censo_dir = os.path.join(mol_dir, "censo")
    create_dir(censo_dir, logger)

    # Obtain conformer ensemble from CREST output
    crest_conformers_file = os.path.join(mol_dir, "crest", "crest_conformers.xyz")
    conformer_ensemble = EnsembleData(input_file=crest_conformers_file)

    # Setting config options
    Config.NCORES = config["N_CORES"]
    Prescreening.set_general_settings(config["censo"]["general"])
    Prescreening.set_settings(config["censo"]["prescreening"])
    Screening.set_settings(config["censo"]["screening"])
    Optimization.set_settings(config["censo"]["optimization"])
    Refinement.set_settings(config["censo"]["refinement"])

    censo_pipeline = [Prescreening, Screening, Optimization, Refinement]
    results, timings = zip(*[part.run(conformer_ensemble) for part in censo_pipeline])


def run_crest_conformer(
    smiles: str,
    mol_type: Literal["monomer", "initiator", "polymer"],
    dir_name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
):
    """
    Running the CREST standard procedure to generate a conformer/rotamer ensemble.
    """
    logger.debug(
        f"Generating a conformer ensemble for {mol_type} ({smiles}) using CREST"
    )
    # Create output dirs
    mol_dir = os.path.join(args.output, "step2", dir_name)
    create_dir(mol_dir, logger)
    pre_dir = os.path.join(mol_dir, "pre")
    create_dir(pre_dir, logger)
    crest_dir = os.path.join(mol_dir, "crest")
    create_dir(crest_dir, logger)

    # Copy most stable conformer from step 1
    xyz_path = copy_most_stable_conformer(dir_name, args, logger)
    xyz_file = Path(xyz_path).name

    # specify location for crest log file
    crest_output_file = os.path.join(crest_dir, "crest.txt")

    # conformer generation with CREST

    crest_args = [
        "crest",
        f"../pre/{xyz_file}",
        "--prop",
        "hess",
        "--prop",
        "autoIR",
    ] + parse_command_args(command_type="crest", config=config, logger=logger)

    try:
        with open(crest_output_file, "w") as fstream:
            subprocess.Popen(
                args=crest_args, stdout=fstream, stderr=subprocess.STDOUT, cwd=post_dir
            ).wait()
    except:
        raise


def main(args, config, logger):
    # get input monomer SMILES strings
    logger.info("Attempting to read input monomer smiles")
    monomer_smiles_list = parse_monomer_smiles(args, logger)
    logger.info("Finished reading input monomer smiles")

    # setup step 2 dir
    step2_dir = os.path.join(args.output, "step2")
    create_dir(step2_dir, logger)

    # run CREST for initiator if ROR reaction
    logger.info("Running CREST for initiator for ROR reaction")
    if config["reaction"]["type"] == "ROR":
        initiator_smiles = config["initiator"]["smiles"]
        initiator_dir_name = os.path.join("initiator", initiator_smiles)
        try:
            run_crest_conformer(
                smiles=initiator_smiles,
                mol_type="initiator",
                dir_name=initiator_dir_name,
                args=args,
                config=config,
                logger=logger,
            )
        except Exception as err:
            logger.exception(err)
            logger.error("Failed to run CREST for initiator")
            raise SystemExit(1)
        else:
            logger.info("Finished running CREST for initiator")


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser = CRESTParser()
    args = parser.parse_args()

    # load logging config from config dir
    try:
        logger = get_logger("step2", args)
    except Exception:
        raise SystemExit(1)
    else:
        logger.info("Logging config loaded")

    # load step 1 config
    try:
        config = get_config(args, logger)
    except Exception:
        raise SystemExit(1)
    else:
        logger.info("Step 2 config loaded")

    # TODO: write assertions for config

    main(args, config, logger)
