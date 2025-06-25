import argparse
import logging
import os
from pathlib import Path
from typing import Any, Literal

from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from rdkit.Chem.rdchem import BondType, EditableMol, Mol
from rdkit.Chem.rdmolfiles import MolFromSmarts, MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import MolzipLabel, MolzipParams, molzip

from larest.constants import HARTTREE_TO_JMOL, INITIATOR_GROUPS, MONOMER_GROUPS


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


def create_pre_post_dirs(dir_path: str | Path, logger: logging.Logger) -> None:
    create_dir(dir_path, logger)
    pre_dir = os.path.join(dir_path, "pre")
    create_dir(pre_dir, logger)
    post_dir = os.path.join(dir_path, "post")
    create_dir(post_dir, logger)


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
    logger.debug(f"Reading monomer smiles from {input_file}")

    try:
        with open(input_file, "r") as fstream:
            monomer_smiles = fstream.read().splitlines()
            for i, smiles in enumerate(monomer_smiles):
                logger.debug(f"Read monomer {i}: {smiles}")

        logger.debug(f"Input monomer smiles: {monomer_smiles}")
        return monomer_smiles
    except Exception as e:
        logger.exception(e)
        logger.error("Failed to read input monomer smiles")
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
                    # FIX: need to extract correct numbers from xtb
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


def get_polymer_unit(
    smiles: str,
    mol_type: Literal["monomer", "initiator"],
    front_dummy: str,
    back_dummy: str,
    logger: logging.Logger,
) -> Mol | None:
    # NOTE: takes the first found match (does not support molecules with >1 possible
    # functional group at the moment)

    if mol_type == "monomer":
        functional_groups = [MolFromSmarts(fg) for fg in MONOMER_GROUPS]
    else:
        functional_groups = [MolFromSmarts(fg) for fg in INITIATOR_GROUPS]

    # convert monomer/initiator smiles and front/back dummies to Mols
    logger.debug(f"Getting {mol_type} unit for polymer construction (smiles: {smiles})")
    logger.debug(f"Front dummy: {front_dummy}, Back dummy {back_dummy}")
    mol = MolFromSmiles(smiles)
    front_dummy = MolFromSmiles(f"[{front_dummy}]").GetAtomWithIdx(0)
    back_dummy = MolFromSmiles(f"[{back_dummy}]").GetAtomWithIdx(0)

    # obtaining the atom ids of the functional group in the monomer/initiator
    fg_atom_idx = []
    for fg in functional_groups:
        if mol.HasSubstructMatch(fg):
            for substruct in mol.GetSubstructMatches(fg):
                for atom_id in substruct:
                    if mol_type == "monomer":
                        if mol.GetAtomWithIdx(atom_id).IsInRing():
                            fg_atom_idx.append(atom_id)
                    else:
                        fg_atom_idx.append(atom_id)
                if len(fg_atom_idx) != 0:
                    logger.debug(f"Functional group detected: {MolToSmiles(fg)}")
                    logger.debug(f"Functional group atom ids: {fg_atom_idx}")
                    break
        if len(fg_atom_idx) != 0:
            break

    # no functional group detected in monomer/initiator
    if len(fg_atom_idx) == 0:
        logger.error(f"No functional group atom ids found for {mol_type} {smiles}")
        return None

    # creating an editable Mol and breaking fg bond
    emol = EditableMol(mol)
    emol.RemoveBond(*fg_atom_idx)

    # adding front dummy to first fg group atom
    emol.AddBond(
        beginAtomIdx=fg_atom_idx[0],
        endAtomIdx=emol.AddAtom(front_dummy),
        order=BondType.SINGLE,
    )
    # adding back dummy to second fg group atom
    emol.AddBond(
        beginAtomIdx=fg_atom_idx[1],
        endAtomIdx=emol.AddAtom(back_dummy),
        order=BondType.SINGLE,
    )

    logger.debug(f"Created polymer unit {MolToSmiles(emol.GetMol())}")
    return emol.GetMol()


def build_polymer(
    monomer_smiles: str,
    length: int,
    reaction_type: Literal["ROR", "RER"],
    config: dict[str, Any],
    logger: logging.Logger,
) -> Mol | None:
    if length <= 1:
        logger.error(f"Please specify a polymer length > 1 (current length: {length}")
        return None

    mol_zip_params = MolzipParams()
    mol_zip_params.label = MolzipLabel.AtomType

    logger.info(f"Attempting to build {reaction_type} polymer")
    if reaction_type == "ROR":
        logger.info(
            f"Monomer smiles: {monomer_smiles}, initiator smiles: {config['initiator']['smiles']}, length: {length}"
        )
    else:
        logger.info(f"Monomer smiles: {monomer_smiles}, length: {length}")

    polymer = get_polymer_unit(
        smiles=monomer_smiles,
        mol_type="monomer",
        front_dummy="Xe",
        back_dummy="Y",
        logger=logger,
    )
    if polymer is None:
        logger.error(f"Failed to create monomer unit from {monomer_smiles}")
        return None

    monomer_units = 1
    front_dummy, back_dummy = "Y", "Zr"
    if reaction_type == "RER":
        length -= 1
    while monomer_units < length:
        repeating_unit = get_polymer_unit(
            smiles=monomer_smiles,
            mol_type="monomer",
            front_dummy=front_dummy,
            back_dummy=back_dummy,
            logger=logger,
        )
        mol_zip_params.setAtomSymbols([front_dummy])
        polymer = molzip(polymer, repeating_unit, mol_zip_params)
        front_dummy, back_dummy = back_dummy, front_dummy
        monomer_units += 1
        logger.debug(
            f"Polymer chain with length={monomer_units}): {MolToSmiles(polymer)}"
        )

    terminal_config = dict(
        ROR=dict(
            smiles=config["initiator"]["smiles"],
            mol_type="initiator",
            front_dummy=front_dummy,
            back_dummy="Xe",
            logger=logger,
        ),
        RER=dict(
            smiles=monomer_smiles,
            mol_type="monomer",
            front_dummy=front_dummy,
            back_dummy="Xe",
            logger=logger,
        ),
    )

    terminal_unit = get_polymer_unit(**terminal_config[reaction_type])

    if terminal_unit is None:
        logger.error(
            f"Failed to create terminal group from {terminal_config[reaction_type]['smiles']}"
        )
        return None

    mol_zip_params.setAtomSymbols([front_dummy, "Xe"])
    polymer = molzip(polymer, terminal_unit, mol_zip_params)
    logger.info(f"Finished building {reaction_type} polymer: {MolToSmiles(polymer)}")
    return MolToSmiles(polymer)
