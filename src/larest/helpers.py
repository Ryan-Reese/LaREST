import argparse
import logging
import os
from pathlib import Path
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from rdkit.Chem.rdchem import Mol
from typing import Any
from larest.constants import HARTTREE_TO_JMOL


def get_mol(smiles: str) -> Mol:
    """Get an rdkit molecule object from a SMILES string."""
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    return mol


def get_ring_size(smiles: str) -> int | None:
    """Get the size of the largest ring in a molecule."""
    mol = get_mol(smiles)
    ring_info = mol.GetRingInfo()
    n_atoms = mol.GetNumAtoms()
    max_ring_size = max([ring_info.MinAtomRingSize(i) for i in range(n_atoms)])
    if max_ring_size == 0:
        return None
    return max_ring_size


def create_dir(dir_path: str | Path, logger: logging.Logger) -> None:
    # create specified dir
    logger.debug(f"Creating directory: {dir_path}")

    try:
        os.makedirs(dir_path, exist_ok=False)
        logger.debug(f"Directory {dir_path} created")
    except FileExistsError:
        logger.warning(f"Directory {dir_path} already exists")


def get_xtb_args(config: dict[str, Any], logger: logging.Logger) -> list[str]:
    xtb_args = []
    try:
        for k, v in zip(config["xtb"].keys(), config["xtb"].values()):
            xtb_args.append(f"--{k}")
            xtb_args.append(str(v))
        logger.debug(f"Returning xtb args: {xtb_args}")
    except Exception as e:
        logger.exception(e)
        logger.warning(
            f"Failed to parse xtb arguments from dictionary {config}, using default xtb arguments"
        )
    return xtb_args


def parse_monomer_smiles(args: argparse.Namespace, logger: logging.Logger) -> list[str]:

    input_file = os.path.join(args.config, "input.txt")
    logger.info(f"Reading monomer smiles from {input_file}")

    try:
        with open(input_file, "r") as fstream:
            monomer_smiles = fstream.read().splitlines()
            for i, smiles in enumerate(monomer_smiles):
                logger.debug(f"Read monomer {i}: {smiles}")

        logger.debug(f"Input monomer smiles: {monomer_smiles}")
        return monomer_smiles
    except Exception as e:
        logger.exception(e)
        raise SystemExit(1)


def parse_xtb(
    xtb_output_file: str | Path, logger: logging.Logger
) -> tuple[float | None, float | None, float | None]:

    enthalpy, entropy, free_energy = None, None, None

    with open(xtb_output_file, "r") as fstream:
        for line in fstream:
            if "H(0)-H(T)+PV" in line:
                fstream.readline()
                thermo = fstream.readline().split()
                try:
                    enthalpy = float(thermo[2]) * HARTTREE_TO_JMOL
                    entropy = float(thermo[3]) * HARTTREE_TO_JMOL / 298.15
                    free_energy = float(thermo[4]) * HARTTREE_TO_JMOL
                except Exception as e:
                    logger.exception(e)

    if not (enthalpy and entropy and free_energy):
        logger.warning(
            f"Failed to extract data from from {xtb_output_file}, assigning None instead"
        )
    return enthalpy, entropy, free_energy
