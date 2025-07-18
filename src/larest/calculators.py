import argparse
import logging
from pathlib import Path
from typing import Any

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

from larest.base import LarestMol
from larest.chem import get_mol
from larest.output import create_dir


def run_rdkit(
    mol: LarestMol,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    # setup RDKit dir if not present
    rdkit_dir = Path(args.output) / mol.dirname / "rdkit"
    create_dir(rdkit_dir, logger)

    logger.debug(
        "Generating conformers and computing energies using RDKit",
    )
    # ) -> tuple[Mol, list[tuple[int, float]]]:
    """
    Generates conformers from a given SMILES string using RDKit
    """
    rdkit_mol = AddHs(get_mol(mol.smiles, logger))

    logger.debug(f"Generating {n_conformers} conformers for rdkit_mol object")
    conformer_ids = EmbedMultipleConfs(
        rdkit_mol,
        n_conformers,
        useRandomCoords=True,
        randomSeed=config["step1"]["rdkit"]["random_seed"],
        boxSizeMult=config["step1"]["rdkit"]["conformer_box_size"],
        numThreads=config["step1"]["n_cores"],
    )

    logger.debug(f"Optimising the {n_conformers} conformers for rdkit_mol object")
    MMFFOptimizerdkit_moleculeConfs(
        rdkit_mol,
        numThreads=config["step1"]["n_cores"],
        maxIters=config["step1"]["rdkit"]["mmff_iters"],
        mmffVariant=config["step1"]["rdkit"]["mmff"],
    )

    logger.debug("Computing rdkit_molecular properties for MMFF")
    mp = MMFFGetrdkit_moleculeProperties(
        rdkit_mol,
        mmffVariant=config["step1"]["rdkit"]["mmff"],
        mmffVerbosity=int(args.verbose),
    )

    logger.debug(f"Computing energies for the {n_conformers} conformers")
    conformer_energies = sorted(
        [
            (
                cid,
                MMFFGetrdkit_moleculeForceField(mol, mp, confId=cid).CalcEnergy()
                * KCALrdkit_mol_TO_JMOL,
            )
            for cid in conformer_ids
        ],
        key=lambda x: x[1],  # sort by energies
    )
    debug_view = list(zip(*conformer_energies, strict=True))
    debug_view = pd.DataFrame({"cid": debug_view[0], "energy": debug_view[1]})
    debug_view = debug_view.set_index("cid")
    logger.debug(debug_view.to_string())

    logger.debug(f"Aligning the {n_conformers} conformers by their geometries")
    Alignrdkit_molConformers(mol, maxIters=config["step1"]["rdkit"]["align_iters"])

    return rdkit_mol, conformer_energies

    # Create output dirs
    # create_dir(mol_dir, logger)
    # pre_dir = os.path.join(mol_dir, "pre")
    # create_dir(pre_dir, logger)
    # post_dir = os.path.join(mol_dir, "post")
    # create_dir(post_dir, logger)

    # Generate and write conformers in .sdf files
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
        logger.exception(f"Failed to write RDKit conformers to {sdf_file}")
        raise
    else:
        logger.debug("Finished writing conformers and their energies")

    # Write conformers to .xyz files
    logger.debug(f"Getting conformer coordinates from {sdf_file}")
    sdfstream = open(sdf_file, "rb")
    mol_supplier = ForwardSDMolSupplier(
        fileobj=sdfstream,
        sanitize=False,
        removeHs=False,
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
                precision=config["step1"]["rdkit"]["precision"],
            )
        except Exception as err:
            logger.exception(err)
            logger.exception(f"Failed to write conformer coordinates to {xyz_file}")
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
            "--namespace",
            f"conformer_{cid}",
            "--parallel",
            f"{config['step1']['n_cores']}",
        ] + parse_command_args(
            command_type="xtb_thermo",
            config=config,
            logger=logger,
        )

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
            logger.exception(
                f"Failed to run xTB command with arguments {xtb_args} in {conformer_dir}",
            )
        else:
            logger.debug(
                f"Finished running xTB on conformer {cid} with output saved to {xtb_output_file}",
            )
    sdfstream.close()
    logger.debug("Finished running xTB on conformers")

    logger.debug("Compiling results of xTB computations")
    results = {
        "conformer_id": [],
        "enthalpy": [],
        "entropy": [],
        "free_energy": [],
        "total_energy": [],
    }

    with os.scandir(post_dir) as folder:
        conformer_dirs = [d for d in folder if d.is_dir()]
    logger.debug(f"Searching conformer dirs {conformer_dirs}")

    for conformer_dir in conformer_dirs:
        xtb_output_file = os.path.join(conformer_dir, f"{conformer_dir.name}.txt")

        try:
            enthalpy, entropy, free_energy, total_energy = parse_xtb_output(
                xtb_output_file,
                config,
                logger,
            )
        except Exception:
            logger.exception(
                f"Failed to parse xtb results for conformer in {conformer_dir.name}",
            )
            continue
        else:
            results["conformer_id"].append(conformer_dir.name.split("_")[1])
            results["enthalpy"].append(enthalpy)
            results["entropy"].append(entropy)
            results["free_energy"].append(free_energy)
            results["total_energy"].append(total_energy)

    results_file = os.path.join(post_dir, "results.csv")
    logger.debug(f"Writing results to {results_file}")

    results = pd.DataFrame(results, index=None, dtype=np.float64).sort_values(
        "free_energy",
    )
    results.to_csv(results_file, header=True, index=False)

    logger.debug(f"Finished writing results for {mol_type} ({smiles})")
