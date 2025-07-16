import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pandas._libs.tslibs.fields import build_isocalendar_sarray

from larest.helpers.chem import build_polymer
from larest.helpers.output import (
    create_dir,
    slugify,
)
from larest.helpers.parsers import (
    CRESTParser,
    parse_command_args,
    parse_most_stable_conformer,
)
from larest.helpers.setup import create_censorc, get_config, get_logger


def run_xtb_thermo(
    smiles: str,
    mol_type: Literal["monomer", "initiator", "polymer"],
    dir_name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Running xTB to compute thermodynamic parameters of CENSO-refined conformers
    """
    logger.debug(f"Running xTB thermo for {mol_type} ({smiles})")

    # Create output dir
    mol_dir = os.path.join(args.output, "step2", dir_name)
    xtb_thermo_dir = os.path.join(mol_dir, "xtb_thermo")
    create_dir(xtb_thermo_dir, logger)

    # specify location for xtb log file
    xtb_output_file = os.path.join(xtb_thermo_dir, "xtb.txt")

    # running thermo calculations with xTB

    xtb_config_file = os.path.join(args.config, "temp", ".xtb.inp")

    xtb_args = [
        "xtb",
        # "--input",
        # f"{os.path.abspath(xtb_config_file)}",
        "thermo",
        "--sthr",
        "100",
        "--temp",
        "298.15",
        "../censo/2_OPTIMIZATION/CONF1/xtb_opt/xtb_opt.xyz",
        "../censo/2_OPTIMIZATION/CONF1/xtb_rrho/hessian",
        # "--namespace",
        # "3_REFINEMENT",
        # "--parallel",
        # f"{config['step2']['n_cores']}",
    ] + parse_command_args(
        command_type="xtb_thermo",
        config=config,
        logger=logger,
    )

    try:
        with open(xtb_output_file, "w") as fstream:
            subprocess.Popen(
                args=xtb_args,
                stdout=fstream,
                stderr=subprocess.STDOUT,
                cwd=xtb_thermo_dir,
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
) -> None:
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
    censo_config_file = os.path.join(args.config, "temp", ".censo2rc")

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


def run_confgen(
    smiles: str,
    mol_type: Literal["monomer", "initiator", "polymer"],
    dir_name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Running the CREST standard procedure to generate a conformer/rotamer ensemble.
    Subsequently performing thermo calculations using xTB on best conformer.
    """
    logger.debug(
        f"Generating a conformer ensemble for {mol_type} ({smiles}) using CREST",
    )
    # Create output dirs
    mol_dir = os.path.join(args.output, "step2", dir_name)
    create_dir(mol_dir, logger)
    pre_dir = os.path.join(mol_dir, "pre")
    create_dir(pre_dir, logger)
    confgen_dir = os.path.join(mol_dir, "confgen")
    create_dir(confgen_dir, logger)
    confgen_crest_dir = os.path.join(mol_dir, "confgen", "crest")
    create_dir(confgen_crest_dir, logger)
    confgen_xtb_dir = os.path.join(mol_dir, "confgen", "xtb")
    create_dir(confgen_xtb_dir, logger)

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
    crest_output_file = os.path.join(confgen_crest_dir, "crest.txt")
    # specify location for xTB log file
    xtb_output_file = os.path.join(confgen_xtb_dir, "xtb.txt")

    # conformer generation with CREST

    crest_default_args = [
        "crest",
        f"../../pre/{xyz_file}",
        "--T",
        f"{config['step2']['n_cores']}",
        "--prop",
        "hess",
        "--prop",
        "autoIR",
    ]

    crest_custom_args = parse_command_args(
        command_type="crest_confgen",
        config=config,
        logger=logger,
    )

    try:
        with open(crest_output_file, "w") as fstream:
            subprocess.Popen(
                args=crest_default_args + crest_custom_args,
                stdout=fstream,
                stderr=subprocess.STDOUT,
                cwd=confgen_crest_dir,
            ).wait()
    except:
        raise
    else:
        logger.debug(
            f"Finished running CREST",
        )

    logger.debug(
        f"Calculating thermodynamic parameters using xTB",
    )

    xtb_default_args = [
        "xtb",
        "../crest/crest_best.xyz",
        "--namespace",
        "crest_best",
        "--parallel",
        f"{config['step2']['n_cores']}",
    ]
    xtb_custom_args = parse_command_args(
        command_type="confgen_xtb",
        config=config,
        logger=logger,
    )

    try:
        with open(xtb_output_file, "w") as fstream:
            subprocess.Popen(
                args=xtb_default_args + xtb_custom_args,
                stdout=fstream,
                stderr=subprocess.STDOUT,
                cwd=confgen_xtb_dir,
            ).wait()
    except:
        raise
    else:
        logger.debug(
            f"Finished xTB",
        )


def main(
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    # setup step 2 dir
    step2_dir = os.path.join(args.output, "step2")
    create_dir(step2_dir, logger)

    # run CREST for initiator if ROR reaction
    logger.info("Running CREST for initiator for ROR reaction")
    if config["reaction"]["type"] == "ROR":
        # initiator_smiles = config["reaction"]["initiator"]
        # initiator_dir_name = os.path.join("initiator", slugify(initiator_smiles))
        monomer_smiles = config["reaction"]["monomers"][0]
        polymer_length = config["reaction"]["lengths"][0]
        polymer_smiles = build_polymer(
            monomer_smiles=monomer_smiles,
            polymer_length=polymer_length,
            reaction_type=config["reaction"]["type"],
            config=config,
            logger=logger,
        )
        polymer_dir_name = os.path.join(
            "polymer",
            f"{slugify(monomer_smiles)}_{polymer_length}",
        )

        try:
            run_crest_confgen(
                smiles=polymer_smiles,
                mol_type="polymer",
                dir_name=polymer_dir_name,
                args=args,
                config=config,
                logger=logger,
            )
            # run_censo(
            #     smiles=polymer_smiles,
            #     mol_type="polymer",
            #     dir_name=polymer_dir_name,
            #     args=args,
            #     config=config,
            #     logger=logger,
            # )
            # run_xtb_thermo(
            #     smiles=initiator_smiles,
            #     mol_type="initiator",
            #     dir_name=initiator_dir_name,
            #     args=args,
            #     config=config,
            #     logger=logger,
            # )
        except Exception as err:
            logger.exception(err)
            logger.exception("Failed to run step 2 for initiator")
            raise SystemExit(1) from err
        else:
            logger.info("Finished running CREST for initiator")

    for monomer_smiles in config["reaction"]["monomers"]:
        pass


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser = CRESTParser()
    args = parser.parse_args()

    # load logging config from config dir
    try:
        logger = get_logger("step2", args)
    except Exception as err:
        raise SystemExit(1) from err
    else:
        logger.info("Logging config loaded")

    # load LaREST config
    try:
        config = get_config(args, logger)
        # create temporary .censo2rc file
        create_censorc(args, logger)
    except Exception as err:
        raise SystemExit(1) from err
    else:
        logger.info("Step 2 config loaded")
        logger.debug(f"Reaction config: {config['reaction']}")
        logger.debug(f"Step 2 config: {config['step2']}")

    # TODO: write assertions for config

    main(args, config, logger)
