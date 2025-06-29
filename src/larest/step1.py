import argparse
import logging
import logging.config
import os
import subprocess
import tomllib
from typing import Any, Literal

import numpy as np
import pandas as pd
from rdkit.Chem.AllChem import MMFFGetMoleculeForceField, MMFFGetMoleculeProperties
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMoleculeConfs
from rdkit.Chem.rdMolAlign import AlignMolConformers
from rdkit.Chem.rdmolfiles import (
    ForwardSDMolSupplier,
    MolFromSmiles,
    MolToXYZFile,
    SDWriter,
)
from rdkit.Chem.rdmolops import AddHs
from tqdm import tqdm

from larest.config.parsers import XTBParser
from larest.constants import KCALMOL_TO_JMOL
from larest.helpers import (
    build_polymer,
    create_dir,
    get_xtb_args,
    parse_monomer_smiles,
    parse_xtb,
)


def generate_conformer_energies(
    smiles: str,
    n_conformers: int,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[Mol, list[tuple[int, float]]]:
    """
    Generates conformers from a given SMILES string using RDKit
    """
    logger.debug(f"Generating RDKit Mol object from SMILES ({smiles})")
    mol = AddHs(MolFromSmiles(smiles), addCoords=True)

    logger.debug(f"Generating {n_conformers} conformers for Mol object")
    conformer_ids = EmbedMultipleConfs(
        mol,
        n_conformers,
        randomSeed=config["rdkit"]["random_seed"],
        useRandomCoords=True,
        boxSizeMult=config["rdkit"]["conformer_box_size"],
        numThreads=config["rdkit"]["threads"],
    )

    logger.debug(f"Optimising the {n_conformers} conformers for Mol object")
    MMFFOptimizeMoleculeConfs(
        mol,
        numThreads=config["rdkit"]["threads"],
        maxIters=config["rdkit"]["mmff_iters"],
        mmffVariant=config["rdkit"]["mmff"],
    )

    logger.debug("Computing molecular properties for MMFF")
    mp = MMFFGetMoleculeProperties(
        mol, mmffVariant=config["rdkit"]["mmff"], mmffVerbosity=int(args.verbose)
    )

    logger.debug(f"Computing energies for the {n_conformers} conformers")
    conformer_energies = sorted(
        [
            (
                cid,
                MMFFGetMoleculeForceField(mol, mp, confId=cid).CalcEnergy()
                * KCALMOL_TO_JMOL,
            )
            for cid in conformer_ids
        ],
        key=lambda x: x[1],  # sort by energies
    )
    debug_view = list(zip(*conformer_energies))
    debug_view = pd.DataFrame(dict(cid=debug_view[0], energy=debug_view[1]))
    debug_view = debug_view.set_index("cid")
    logger.debug(debug_view.to_string())

    logger.debug(f"Aligning the {n_conformers} conformers by their geometries")
    AlignMolConformers(mol, maxIters=config["rdkit"]["align_iters"])

    return mol, conformer_energies


def run_xtb_conformers(
    smiles: str,
    n_conformers: int,
    mol_type: Literal["monomer", "initiator", "polymer"],
    dir_name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
):
    """
    Calculates the thermodynamic parameters of the molecule with the specified SMILES string
    """

    logger.debug(
        f"Genering conformers and computing their energies of {mol_type} ({smiles}) using RDKit"
    )

    # Create output dirs
    mol_dir = os.path.join(args.output, "step1", f"{dir_name}")
    create_dir(mol_dir, logger)
    pre_dir = os.path.join(mol_dir, "pre")
    create_dir(pre_dir, logger)
    post_dir = os.path.join(mol_dir, "post")
    create_dir(post_dir, logger)

    # Generate and write conformers in .sdf files
    logger.debug("Generating conformers and their energies using RDKit")
    mol, conformer_energies = generate_conformer_energies(
        smiles=StandardizeSmiles(smiles),  # USES CANONICAL SMILES
        n_conformers=n_conformers,
        args=args,
        config=config,
        logger=logger,
    )
    logger.debug("Finished generating conformers")

    # Write conformers to .sdf file
    sdf_file = os.path.join(pre_dir, "conformers.sdf")
    logger.debug(f"Writing conformers and their energies to {sdf_file}")

    try:
        with open(sdf_file, "w") as fstream:
            writer = SDWriter(fstream)
            for cid, energy in conformer_energies:
                mol.SetIntProp("conformer_id", cid)
                mol.SetDoubleProp("energy", energy)
                writer.write(mol, confId=cid)
            writer.close()
    except Exception as err:
        logger.exception(err)
        logger.error(f"Failed to write RDKit conformers to {sdf_file}")
        raise
    else:
        logger.debug("Finished writing conformers and their energies")

    # Write conformers to .xyz files
    logger.debug(f"Getting conformer coordinates from {sdf_file}")
    sdfstream = open(sdf_file, "rb")
    mol_supplier = ForwardSDMolSupplier(
        fileobj=sdfstream, sanitize=False, removeHs=False
    )

    # Iterating over conformers
    logger.debug("Computing thermodynamic parameters of conformers using xTB")
    for conformer in tqdm(
        mol_supplier,
        desc="iterating over conformers",
        total=n_conformers,
    ):
        # Conformer id and location of xyz file
        cid = conformer.GetIntProp("conformer_id")
        xyz_file = os.path.join(pre_dir, f"conformer_{cid}.xyz")

        logger.debug(f"Writing conformer {cid} coordinates to {xyz_file}")
        try:
            MolToXYZFile(
                mol=mol,
                filename=xyz_file,
                precision=config["rdkit"]["precision"],
            )
        except Exception as err:
            logger.exception(err)
            logger.error(f"Failed to write conformer coordinates to {xyz_file}")
            raise
        else:
            logger.debug(f"Finished writing conformers {cid} coordinates")

        # Creating output dir for conformer post-xtb
        conformer_dir = os.path.join(post_dir, f"conformer_{cid}")
        create_dir(conformer_dir, logger)

        # Specify location for xtb log file
        xtb_output_file = os.path.join(conformer_dir, f"conformer_{cid}.txt")

        # Optimisation with xTB
        xtb_args = [
            "xtb",
            f"../../pre/conformer_{cid}.xyz",
            "--verbose",
            "--namespace",
            f"conformer_{cid}",
            "--json",
        ] + get_xtb_args(config, logger)

        logger.debug(f"Running xTB on conformer {cid}")
        try:
            with open(xtb_output_file, "w") as fstream:
                subprocess.Popen(
                    args=xtb_args,
                    stdout=fstream,
                    stderr=subprocess.STDOUT,
                    cwd=conformer_dir,
                ).wait()
        except Exception as err:
            logger.exception(err)
            logger.error(
                f"Failed to run xTB command with arguments {xtb_args} in {conformer_dir}"
            )
        else:
            logger.debug(
                f"Finished running xTB on conformer {cid} with output saved to {xtb_output_file}"
            )
    sdfstream.close()
    logger.debug("Finished running xTB on conformers")

    logger.debug("Compiling results of xTB computations")
    results = dict(
        conformer_id=[], enthalpy=[], entropy=[], free_energy=[], total_energy=[]
    )

    with os.scandir(post_dir) as folder:
        conformer_dirs = [d for d in folder if d.is_dir()]
    logger.debug(f"Searching conformer dirs {conformer_dirs}")

    for conformer_dir in conformer_dirs:
        xtb_output_file = os.path.join(conformer_dir, f"{conformer_dir.name}.txt")

        try:
            enthalpy, entropy, free_energy, total_energy = parse_xtb(
                xtb_output_file, config, logger
            )
        except Exception:
            continue
        else:
            results["conformer_id"].append(conformer_dir.name.split("_")[1])
            results["enthalpy"].append(enthalpy)
            results["entropy"].append(entropy)
            results["free_energy"].append(free_energy)
            results["total_energy"].append(total_energy)

    results_file = os.path.join(post_dir, "results.csv")
    logger.debug(f"Writing results to {results_file}")

    results = pd.DataFrame(results, dtype=np.float64).sort_values("free_energy")
    results.to_csv(results_file, header=True, index=False)

    logger.debug(f"Finished writing results for {mol_type} ({smiles})")


def compile_results(monomer_smiles, args, config, logger):
    results = dict()
    step1_dir = os.path.join(args.output, "step1")
    monomer_dir = os.path.join(step1_dir, f"monomer_{monomer_smiles}")
    monomer_results_file = os.path.join(monomer_dir, "post", "results.csv")
    monomer_results = pd.read_csv(
        monomer_results_file, header=0, index_col="conformer_id", dtype=np.float64
    )
    results["monomer"] = monomer_results.iloc[0].to_dict()

    with os.scandir(step1_dir) as folder:
        polymer_dirs = [
            d
            for d in folder
            if (d.is_dir() and (f"polymer_{monomer_smiles}" in d.name))
        ]

    for polymer_dir in polymer_dirs:
        polymer_length = polymer_dir.name.split("_")[2]
        polymer_results_file = os.path.join(polymer_dir, "post", "results.csv")
        polymer_results = pd.read_csv(
            polymer_results_file, header=0, index_col="conformer_id", dtype=np.float64
        )
        if "polymer" in results:
            results["polymer"][polymer_length] = polymer_results.iloc[0].to_dict()
        else:
            results["polymer"] = dict()
            results["polymer"][polymer_length] = polymer_results.iloc[0].to_dict()

    compiled_results = dict(polymer_length=[], monomer_enthalpy=[], polymer_enthalpy=[])

    for polymer_length in results["polymer"].keys():
        compiled_results["polymer_length"].append(int(polymer_length))
        compiled_results["monomer_enthalpy"].append(results["monomer"]["enthalpy"])
        compiled_results["polymer_enthalpy"].append(
            results["polymer"][polymer_length]["enthalpy"]
        )

    compiled_results_file = os.path.join(step1_dir, f"results_{monomer_smiles}.csv")

    compiled_results = pd.DataFrame(compiled_results).sort_values("polymer_length")
    compiled_results["unit_polymer_enthalpy"] = (
        compiled_results["polymer_enthalpy"] / compiled_results["polymer_length"]
    )
    compiled_results["delta_h"] = (
        compiled_results["unit_polymer_enthalpy"] - compiled_results["monomer_enthalpy"]
    )

    compiled_results.to_csv(compiled_results_file, header=True, index=False)


def main(args, config, logger):
    # get input monomer SMILES strings
    logger.info("Attempting to read input monomer smiles")
    monomer_smiles_list = parse_monomer_smiles(args, logger)
    logger.info("Finished reading input monomer smiles")

    # setup output dir
    output_dir = os.path.join(args.output, "step1")
    create_dir(output_dir, logger)

    # run xtb for initiator if ROR reaction
    logger.info("Running xTB for initiator for ROR reaction")
    if config["reaction"]["type"] == "ROR":
        initiator_smiles = config["initiator"]["smiles"]
        initiator_n_conformers = config["initiator"]["n_conformers"]
        initiator_dir_name = os.path.join("initiator", initiator_smiles)
        try:
            run_xtb_conformers(
                smiles=initiator_smiles,
                n_conformers=initiator_n_conformers,
                mol_type="initiator",
                dir_name=initiator_dir_name,
                args=args,
                config=config,
                logger=logger,
            )
        except Exception:
            logger.error("Failed to run xTB for initiator")
            raise SystemExit(1)
        else:
            logger.info("Finished running xTB for initiator")

    # iterate over monomers in input list
    for monomer_smiles in tqdm(monomer_smiles_list, desc="Running xTB for monomers"):
        # TODO: need to decide how many conformers to generate
        # ring_size = get_ring_size(smiles)
        # logger.debug(f"Computed ring size: {ring_size}")

        # run xtb for monomer
        monomer_n_conformers = config["monomer"]["n_conformers"]
        monomer_dir_name = os.path.join("monomer", monomer_smiles)

        run_xtb_conformers(
            smiles=monomer_smiles,
            n_conformers=monomer_n_conformers,
            mol_type="monomer",
            dir_name=monomer_dir_name,
            args=args,
            config=config,
            logger=logger,
        )

        # iterate over polymer lengths
        for polymer_length in tqdm(
            config["reaction"]["lengths"], desc="Running xTB for each polymer length"
        ):
            # build polymer
            try:
                polymer_smiles = build_polymer(
                    monomer_smiles=monomer_smiles,
                    polymer_length=polymer_length,
                    reaction_type=config["reaction"]["type"],
                    config=config,
                    logger=logger,
                )
            except Exception:
                pass
            else:
                pass
            if polymer_smiles is None:
                logger.warning(
                    f"Failed to build polymer of length {length}, moving onto next monomer"
                )
                break
            # run xtb for polymer
            run_xtb_conformers(
                smiles=polymer_smiles,
                mol_type="polymer",
                dir_name=f"{smiles}_{length}",
                args=args,
                config=config,
                logger=logger,
            )

        compile_results(monomer_smiles=smiles, args=args, config=config, logger=logger)


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser = XTBParser()
    args = parser.parse_args()

    # load logging config from config dir
    logging_config_file = os.path.join(args.config, "logging.toml")
    try:
        with open(logging_config_file, "rb") as fstream:
            log_config = tomllib.load(fstream)
    except:
        print(f"Failed to load logging config from {logging_config_file}")
        raise
    else:
        log_config["handlers"]["file"]["filename"] = f"{args.output}/larest.log"
        logging.config.dictConfig(log_config)
        logger = logging.getLogger("Step 1")
        logger.info("Logging config loaded")

    # load step 1 config
    step1_config_file = os.path.join(args.config, "step1.toml")

    try:
        with open(step1_config_file, "rb") as fstream:
            config = tomllib.load(fstream)
    except Exception as err:
        logger.exception(err)
        logger.error(f"Failed to load Step 1 config from {step1_config_file}")
        raise SystemExit(1)
    else:
        logger.info("Step 1 config loaded")

    # TODO: write assertions for config

    # main(args, config, logger)
    generate_conformer_energies(
        config["initiator"]["smiles"],
        config["initiator"]["n_conformers"],
        args,
        config,
        logger,
    )
