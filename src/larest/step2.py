import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from larest.helpers.output import (
    copy_most_stable_conformer,
    create_dir,
    slugify,
)
from larest.helpers.parsers import (
    CRESTParser,
    parse_command_args,
    parse_monomer_smiles,
    parse_most_stable_conformer,
)
from larest.helpers.setup import get_config, get_logger


def run_crest_thermo(
    smiles: str,
    mol_type: Literal["monomer", "initiator", "polymer"],
    dir_name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
):
    """
    Running CREST to compute thermodynamic parameters of CENSO-refined conformers
    """
    logger.debug(f"Running CREST thermo for {mol_type} ({smiles})")

    # Create output dir
    mol_dir = os.path.join(args.output, "step2", dir_name)
    crest_thermo_dir = os.path.join(mol_dir, "crest_thermo")
    create_dir(crest_thermo_dir, logger)

    # specify location for crest log file
    crest_output_file = os.path.join(crest_thermo_dir, "crest.txt")

    # running DFT refinement with CENSO
    censo_config_file = os.path.join(args.config, ".censo2rc")

    censo_args = [
        "censo",
        "--input",
        "../crest_conf/crest_conformers.xyz",
        "--inprc",
        f"{os.path.abspath(censo_config_file)}",
        "--maxcores",
        f"{config['step2']['n_cores']}",
    ]

    try:
        with open(censo_output_file, "w") as fstream:
            subprocess.Popen(
                args=censo_args,
                stdout=fstream,
                stderr=subprocess.STDOUT,
                cwd=censo_dir,
            ).wait()
    except:
        raise


def run_censo(
    smiles: str,
    mol_type: Literal["monomer", "initiator", "polymer"],
    dir_name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
):
    """
    Running CENSO to DFT refine the conformer ensemble from CREST.
    """
    logger.debug(f"Running the CENSO pipeline for {mol_type} ({smiles})")

    # Create output dir
    mol_dir = os.path.join(args.output, "step2", dir_name)
    censo_dir = os.path.join(mol_dir, "censo")
    create_dir(censo_dir, logger)

    # specify location for censo log file
    censo_output_file = os.path.join(censo_dir, "censo.log")

    # running DFT refinement with CENSO
    censo_config_file = os.path.join(args.config, ".censo2rc")

    censo_args = [
        "censo",
        "--input",
        "../crest_conf/crest_conformers.xyz",
        "--inprc",
        f"{os.path.abspath(censo_config_file)}",
        "--maxcores",
        f"{config['censo']['general']['maxcores']}",
    ]

    try:
        with open(censo_output_file, "w") as fstream:
            subprocess.Popen(
                args=censo_args,
                stdout=fstream,
                stderr=subprocess.STDOUT,
                cwd=censo_dir,
            ).wait()
    except:
        raise


def run_crest_confgen(
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
    crest_conf_dir = os.path.join(mol_dir, "crest_conf")
    create_dir(crest_conf_dir, logger)

    # Copy most stable conformer from step 1
    step1_mol_dir = os.path.join(args.output, "step1", dir_name)
    conformer_id = int(parse_most_stable_conformer(step1_mol_dir)["conformer_id"])
    conformer_xyz_file = os.path.join(
        step1_mol_dir,
        "post",
        f"conformer_{conformer_id}",
        f"conformer_{conformer_id}.xtbopt.xyz",
    )
    xyz_file = Path(shutil.copy2(conformer_xyz_file, pre_dir)).name

    # specify location for crest log file
    crest_output_file = os.path.join(crest_conf_dir, "crest.txt")

    # conformer generation with CREST

    crest_args = [
        "crest",
        f"../pre/{xyz_file}",
        "--T",
        f"{config['step2']['n_cores']}",
        "--prop",
        "hess",
        "--prop",
        "autoIR",
    ] + parse_command_args(
        command_type="crest_confgen",
        config=config,
        logger=logger,
    )

    try:
        with open(crest_output_file, "w") as fstream:
            subprocess.Popen(
                args=crest_args,
                stdout=fstream,
                stderr=subprocess.STDOUT,
                cwd=crest_conf_dir,
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
        initiator_smiles = config["reaction"]["initiator"]
        initiator_dir_name = os.path.join("initiator", slugify(initiator_smiles))
        try:
            run_crest_confgen(
                smiles=initiator_smiles,
                mol_type="initiator",
                dir_name=initiator_dir_name,
                args=args,
                config=config,
                logger=logger,
            )
            run_censo(
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

    for monomer_smiles in monomer_smiles_list:
        pass


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

    # load LaREST config
    try:
        config = get_config(args, logger)
    except Exception:
        raise SystemExit(1)
    else:
        logger.info("Step 2 config loaded")
        logger.debug(f"Reaction config: {config['reaction']}")
        logger.debug(f"Step 2 config: {config['step2']}")

    # TODO: write assertions for config

    main(args, config, logger)
