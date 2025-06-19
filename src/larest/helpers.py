import os
import logging
from pathlib import Path
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from rdkit.Chem.rdchem import Mol


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
    logger.info(f"Creating directory: {dir_path}")

    try:
        os.mkdir(dir_path)
        logger.info(f"Directory {dir_path} created")
    except FileExistsError:
        logger.warning(f"Directory {dir_path} already exists")
        os.makedirs(dir_path, exist_ok=True)
