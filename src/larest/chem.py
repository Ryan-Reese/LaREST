from logging import Logger
from typing import Any, Literal

from rdkit.Chem.rdchem import Atom, BondType, EditableMol, Mol, RingInfo
from rdkit.Chem.rdmolfiles import MolFromSmarts, MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import (
    AddHs,
    MolzipLabel,
    MolzipParams,
    RemoveHs,
    RemoveHsParameters,
    molzip,
)

from larest.constants import INITIATOR_GROUPS, MONOMER_GROUPS
from larest.exceptions import PolymerBuildError


def get_mol(smiles: str, logger: Logger) -> Mol:
    """Get an rdkit molecule object from a SMILES string."""
    mol: Mol = MolFromSmiles(smiles)
    if mol is None:
        raise PolymerBuildError(
            f"Failed to create RDKit Mol object from SMILES: {smiles}",
        )
    logger.debug(f"Created RDKit Mol object from SMILES: {smiles}")
    return mol


def get_ring_size(smiles: str, logger: Logger) -> int | None:
    """Get the size of the functional group ring in the monomer."""
    try:
        mol: Mol = get_mol(smiles, logger)
    except PolymerBuildError:
        logger.exception(f"Failed to get ring size for SMILES: {smiles}")
        raise

    ring_info: RingInfo = mol.GetRingInfo()
    functional_groups: list[Mol] = [
        MolFromSmarts(fg_smarts) for fg_smarts in MONOMER_GROUPS.values()
    ]
    for fg_mol in functional_groups:
        if mol.HasSubstructMatch(fg_mol):
            substruct_atom_idx: tuple[int] = mol.GetSubstructMatch(fg_mol)
            for atom_id in substruct_atom_idx:
                substruct_atom: Atom = mol.GetAtomWithIdx(atom_id)
                if (substruct_atom.GetAtomicNum() != 6) and (substruct_atom.IsInRing()):
                    ring_size: int = ring_info.MinAtomRingSize(atom_id)
                    logger.debug(f"Found ring size {ring_size} for SMILES: {smiles}")
                    return ring_size
    logger.warning(f"Failed to detect ring size for SMILES: {smiles}")
    return None


def get_polymer_unit(
    smiles: str,
    mol_type: Literal["monomer", "initiator"],
    front_dummy: str,
    back_dummy: str,
    logger: Logger,
) -> Mol:
    # NOTE: output for molecules with >1 ring-opening functional group
    # is deterministic but not yet customisable

    match mol_type:
        case "monomer":
            functional_groups: list[Mol] = [
                MolFromSmarts(fg_smarts) for fg_smarts in MONOMER_GROUPS.values()
            ]
        case "initiator":
            functional_groups: list[Mol] = [
                MolFromSmarts(fg_smarts) for fg_smarts in INITIATOR_GROUPS.values()
            ]

    # convert monomer/initiator smiles and front/back dummies to RDKit Mol/Atoms
    logger.debug(f"Getting {mol_type} unit for polymer construction (SMILES: {smiles})")
    logger.debug(f"Front dummy: {front_dummy}, Back dummy {back_dummy}")
    # creating the RDKit mol object
    match mol_type:
        case "monomer":
            mol: Mol = MolFromSmiles(smiles)
        case "initiator":
            # need to add Hs to break O-H bond
            mol: Mol = AddHs(
                mol=MolFromSmiles(smiles),
                explicitOnly=False,
            )
    front_dummy_atom: Atom = MolFromSmiles(f"[{front_dummy}]").GetAtomWithIdx(0)
    back_dummy_atom: Atom = MolFromSmiles(f"[{back_dummy}]").GetAtomWithIdx(0)

    # obtaining the atom ids of the functional group in the monomer/initiator
    fg_atom_idx = []
    for fg_mol in functional_groups:
        if mol.HasSubstructMatch(fg_mol):
            substruct_atom_idx: tuple[int] = mol.GetSubstructMatch(fg_mol)
            for atom_id in substruct_atom_idx:
                substruct_atom: Atom = mol.GetAtomWithIdx(atom_id)
                match mol_type:
                    case "monomer":
                        # break bond between carbonyl C and electrophilic neighbour
                        if substruct_atom.IsInRing() and (
                            (
                                BondType.DOUBLE
                                in [
                                    bond.GetBondType()
                                    for bond in substruct_atom.GetBonds()
                                ]
                            )
                            or (substruct_atom.GetAtomicNum() != 6)
                        ):
                            fg_atom_idx.append(atom_id)
                    case "initiator":
                        # break O-H bond
                        fg_atom_idx.append(atom_id)
                if len(fg_atom_idx) == 2:
                    break
        if len(fg_atom_idx) != 0:
            if len(fg_atom_idx) != 2:
                raise PolymerBuildError(
                    f"Trying to break bond between >2 atoms for functional group {MolToSmiles(fg_mol)}",
                )
            logger.debug(f"Functional group detected: {MolToSmiles(fg_mol)}")
            break

    # no functional group detected in monomer/initiator
    if len(fg_atom_idx) == 0:
        raise PolymerBuildError(
            f"No functional group atom ids found for {mol_type} SMILES: {smiles}",
        )
    logger.debug(f"Functional group atom ids: {fg_atom_idx}")

    # creating an editable Mol
    emol: EditableMol = EditableMol(mol)
    try:
        # breaking functional group bond
        emol.RemoveBond(*fg_atom_idx)
        # adding front dummy to first fg group atom
        emol.AddBond(
            beginAtomIdx=fg_atom_idx[0],
            endAtomIdx=emol.AddAtom(front_dummy_atom),
            order=BondType.SINGLE,
        )
        # adding back dummy to second fg group atom
        emol.AddBond(
            beginAtomIdx=fg_atom_idx[1],
            endAtomIdx=emol.AddAtom(back_dummy_atom),
            order=BondType.SINGLE,
        )
    except Exception as err:
        logger.exception(err)
        logger.exception(f"Failed to edit {mol_type} unit {MolToSmiles(emol.GetMol())}")
        raise
    else:
        logger.debug(f"Created polymer unit {MolToSmiles(emol.GetMol())}")
        return emol.GetMol()


def build_polymer(
    monomer_smiles: str,
    polymer_length: int,
    reaction_type: Literal["ROR", "RER"],
    config: dict[str, Any],
    logger: Logger,
) -> str:
    if polymer_length <= 1 and reaction_type == "RER":
        raise PolymerBuildError(
            f"Please specify a polymer length > 1 for RER reaction (current length: {polymer_length})",
        )
    if polymer_length < 1 and reaction_type == "ROR":
        raise PolymerBuildError(
            f"Please specify a polymer length >= 1 for ROR reaction (current length: {polymer_length}",
        )

    logger.debug(f"Building {reaction_type} polymer")
    logger.debug(f"Monomer smiles: {monomer_smiles}, length: {polymer_length}")
    if reaction_type == "ROR":
        logger.debug(f"Initiator smiles: {config['reaction']['initiator']}")

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
        logger.exception(f"Failed to create monomer unit from SMILES: {monomer_smiles}")
        raise

    monomer_units: int = 1
    front_dummy: str = "Y"
    back_dummy: str = "Zr"

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
            logger.exception(
                f"Failed to zip polymer units together: {MolToSmiles(polymer)} {MolToSmiles(repeating_unit)}",
            )
            raise
        else:
            front_dummy, back_dummy = back_dummy, front_dummy
            monomer_units += 1
            logger.debug(
                f"Polymer chain with length={monomer_units}): {MolToSmiles(polymer)}",
            )

    try:
        match reaction_type:
            case "ROR":
                terminal_unit = get_polymer_unit(
                    smiles=config["reaction"]["initiator"],
                    mol_type="initiator",
                    front_dummy="Xe",
                    back_dummy=front_dummy,
                    logger=logger,
                )
            case "RER":
                terminal_unit = get_polymer_unit(
                    smiles=monomer_smiles,
                    mol_type="monomer",
                    front_dummy=front_dummy,
                    back_dummy="Xe",
                    logger=logger,
                )
    except Exception:
        logger.exception(
            f"Failed to create terminal group for {reaction_type} reaction",
        )
        raise

    mol_zip_params.setAtomSymbols([front_dummy, "Xe"])
    try:
        match reaction_type:
            case "ROR":
                # Need to remove Hs following initiator addition
                polymer = RemoveHs(
                    mol=molzip(polymer, terminal_unit, mol_zip_params),
                    implicitOnly=False,
                    sanitize=True,
                )
            case "RER":
                polymer = molzip(polymer, terminal_unit, mol_zip_params)
    except Exception as err:
        logger.exception(err)
        logger.exception(
            f"Failed to zip polymer and terminal unit together: {MolToSmiles(polymer)} {MolToSmiles(terminal_unit)}",
        )
        raise
    else:
        logger.info(
            f"Finished building {reaction_type} polymer: {MolToSmiles(polymer)}",
        )
    return MolToSmiles(polymer)
