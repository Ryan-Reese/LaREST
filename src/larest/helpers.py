import argparse
import logging
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from rdkit.Chem.rdchem import BondType, EditableMol, Mol
from rdkit.Chem.rdmolfiles import MolFromSmarts, MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import MolzipLabel, MolzipParams, molzip

from larest.constants import HARTTREE_TO_JMOL, INITIATOR_GROUPS, MONOMER_GROUPS
from larest.exceptions import PolymerLengthError, PolymerUnitError


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
    except FileExistsError:
        logger.warning(f"Directory {dir_path} already exists")
    else:
        logger.debug(f"Directory {dir_path} created")


def get_xtb_args(config: dict[str, Any], logger: logging.Logger) -> list[str]:
    xtb_args = []
    try:
        for k, v in zip(config["xtb"].keys(), config["xtb"].values()):
            xtb_args.append(f"--{k}")
            xtb_args.append(str(v))
    except Exception as err:
        logger.exception(err)
        logger.warning(
            f"Failed to parse xtb arguments from dictionary {config['xtb']}, using default xtb arguments"
        )
        return []
    else:
        logger.debug(f"Returning xtb args: {xtb_args}")
        return xtb_args


def parse_monomer_smiles(args: argparse.Namespace, logger: logging.Logger) -> list[str]:
    input_file = os.path.join(args.config, "input.txt")
    logger.debug(f"Reading monomer smiles from {input_file}")

    try:
        with open(input_file, "r") as fstream:
            monomer_smiles = fstream.read().splitlines()
    except Exception as err:
        logger.exception(err)
        logger.error("Failed to read input monomer smiles")
        raise SystemExit(1)
    else:
        for i, smiles in enumerate(monomer_smiles):
            logger.debug(f"Read monomer {i}: {smiles}")
        return monomer_smiles


def parse_xtb(
    xtb_output_file: str | Path, config: dict[str, Any], logger: logging.Logger
) -> tuple[float | None, float | None, float | None, float | None]:
    enthalpy, entropy, free_energy, total_energy = None, None, None, None

    logger.debug(f"Searching for results in file {xtb_output_file}")
    try:
        with open(xtb_output_file, "r") as fstream:
            for i, line in enumerate(fstream):
                if "TOTAL ENERGY" in line:
                    try:
                        total_energy = float(line.split()[3]) * HARTTREE_TO_JMOL
                    except Exception as err:
                        logger.exception(err)
                        logger.error(
                            f"Failed to extract total energy from line {i}: {line}"
                        )
                elif "TOTAL ENTHALPY" in line:
                    try:
                        enthalpy = float(line.split()[3]) * HARTTREE_TO_JMOL
                    except Exception as err:
                        logger.exception(err)
                        logger.error(
                            f"Failed to extract total enthalpy from line {i}: {line}"
                        )
                elif "TOTAL FREE ENERGY" in line:
                    try:
                        free_energy = float(line.split()[4]) * HARTTREE_TO_JMOL
                    except Exception as err:
                        logger.exception(err)
                        logger.error(
                            f"Failed to extract total free energy from line {i}: {line}"
                        )
    except Exception as err:
        logger.exception(err)
        logger.error(f"Failed to parse xtb results from {xtb_output_file}")
        raise
    else:
        if enthalpy and free_energy:
            entropy = (enthalpy - free_energy) / config["xtb"]["etemp"]
        if not (enthalpy and free_energy and total_energy):
            logger.warning(
                f"Failed to extract necessary data from {xtb_output_file}, missing data will be assigned None"
            )
        logger.debug(
            f"Found enthalpy: {enthalpy}, entropy: {entropy}, free_energy: {free_energy}, total energy: {total_energy}"
        )

    return enthalpy, entropy, free_energy, total_energy


def get_polymer_unit(
    smiles: str,
    mol_type: Literal["monomer", "initiator"],
    front_dummy: str,
    back_dummy: str,
    logger: logging.Logger,
) -> Mol:
    # WARNING: does not support molecules with >1 ring-opening functional group

    if mol_type == "monomer":
        functional_groups = [MolFromSmarts(fg) for fg in MONOMER_GROUPS]
    elif mol_type == "initiator":
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
                        # break bond between cyclic atoms
                        if mol.GetAtomWithIdx(atom_id).IsInRing():
                            fg_atom_idx.append(atom_id)
                    else:
                        fg_atom_idx.append(atom_id)
                if len(fg_atom_idx) != 0:
                    logger.debug(f"Functional group detected: {MolToSmiles(fg)}")
                    break
        if len(fg_atom_idx) != 0:
            break

    # no functional group detected in monomer/initiator
    try:
        if len(fg_atom_idx) == 0:
            raise PolymerUnitError(
                f"No functional group atom ids found for {mol_type} {smiles}"
            )
    except PolymerUnitError as err:
        logger.exception(err)
        raise
    else:
        logger.debug(f"Functional group atom ids: {fg_atom_idx}")

    # creating an editable Mol
    emol = EditableMol(mol)
    try:
        # breaking functional group bond
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
    except Exception as err:
        logger.exception(err)
        logger.error(f"Failed to edit {mol_type} unit {MolToSmiles(emol.GetMol())}")
        raise
    else:
        logger.debug(f"Created polymer unit {MolToSmiles(emol.GetMol())}")
        return emol.GetMol()


def build_polymer(
    monomer_smiles: str,
    polymer_length: int,
    reaction_type: Literal["ROR", "RER"],
    config: dict[str, Any],
    logger: logging.Logger,
) -> str:
    try:
        if polymer_length <= 1 and reaction_type == "RER":
            raise PolymerLengthError(
                f"Please specify a polymer length > 1 for RER reaction (current length: {polymer_length})"
            )
        elif polymer_length < 1 and reaction_type == "ROR":
            raise PolymerLengthError(
                f"Please specify a polymer length >= 1 for ROR reaction (current length: {polymer_length}"
            )
    except PolymerLengthError as err:
        logger.exception(err)
        raise

    logger.debug(f"Building {reaction_type} polymer")
    if reaction_type == "ROR":
        logger.info(
            f"Monomer smiles: {monomer_smiles}, initiator smiles: {config['initiator']['smiles']}, length: {polymer_length}"
        )
    else:
        logger.info(f"Monomer smiles: {monomer_smiles}, length: {polymer_length}")

    mol_zip_params = MolzipParams()
    mol_zip_params.label = MolzipLabel.AtomType

    try:
        polymer = get_polymer_unit(
            smiles=monomer_smiles,
            mol_type="monomer",
            front_dummy="Xe",
            back_dummy="Y",
            logger=logger,
        )
    except Exception:
        logger.error(f"Failed to create monomer unit from {monomer_smiles}")
        raise

    monomer_units = 1
    front_dummy, back_dummy = "Y", "Zr"
    if reaction_type == "RER":
        polymer_length -= 1
    while monomer_units < polymer_length:
        repeating_unit = get_polymer_unit(
            smiles=monomer_smiles,
            mol_type="monomer",
            front_dummy=front_dummy,
            back_dummy=back_dummy,
            logger=logger,
        )
        mol_zip_params.setAtomSymbols([front_dummy])
        try:
            polymer = molzip(polymer, repeating_unit, mol_zip_params)
        except Exception as err:
            logger.exception(err)
            logger.error(
                f"Failed to zip polymer units together: {MolToSmiles(polymer)} {MolToSmiles(repeating_unit)}"
            )
            raise
        else:
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

    try:
        terminal_unit = get_polymer_unit(**terminal_config[reaction_type])
    except Exception:
        logger.error(
            f"Failed to create terminal group from {terminal_config[reaction_type]['smiles']}"
        )
        raise

    mol_zip_params.setAtomSymbols([front_dummy, "Xe"])
    try:
        polymer = molzip(polymer, terminal_unit, mol_zip_params)
    except Exception as err:
        logger.exception(err)
        logger.error(
            f"Failed to zip polymer and terminal unit together: {MolToSmiles(polymer)} {MolToSmiles(terminal_unit)}"
        )
        raise
    else:
        logger.info(
            f"Finished building {reaction_type} polymer: {MolToSmiles(polymer)}"
        )
    return MolToSmiles(polymer)


def get_most_stable_conformer(mol_dir: str | os.DirEntry) -> dict[str, float]:
    results_file = os.path.join(mol_dir, "post", "results.csv")

    results = pd.read_csv(
        results_file, header=0, index_col="conformer_id", dtype=np.float64
    )

    # NOTE: The first row corresponds to the most stable conformer (lowest free energy, previously sorted)

    return results.iloc[0].to_dict()
