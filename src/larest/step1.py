import argparse
import logging
import logging.config
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from rdkit.Chem.AllChem import MMFFGetMoleculeForceField, MMFFGetMoleculeProperties
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from rdkit.Chem.rdchem import Atom, Mol
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

from larest.config.parsers import XTBParser
from larest.helpers import (
    build_polymer,
    create_dir,
    create_pre_post_dirs,
    get_ring_size,
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
):
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

    logger.debug(f"Optimising {n_conformers} conformers for Mol object")
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
            (cid, MMFFGetMoleculeForceField(mol, mp, confId=cid).CalcEnergy())
            for cid in conformer_ids
        ],
        key=lambda x: x[1],
    )

    logger.debug(f"Aligning {n_conformers} conformers by their geometries")
    AlignMolConformers(mol, maxIters=config["rdkit"]["align_iters"])

    return mol, conformer_energies


def run_xtb_conformers(
    smiles: str,
    mol_type: Literal["monomer", "initiator", "polymer"],
    dir_name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
):
    """
    Calculates the thermodynamic parameters of the molecule with the specified SMILES string
    """

    logger.info(f"Computing thermodynamic parameters of {mol_type} ({smiles})")

    # Create output dirs
    mol_dir = os.path.join(args.output, "step1", f"{mol_type}_{dir_name}")
    create_dir(mol_dir, logger)
    pre_dir = os.path.join(mol_dir, "pre")
    create_dir(pre_dir, logger)
    post_dir = os.path.join(mol_dir, "post")
    create_dir(post_dir, logger)

    # Generate and write conformers in .sdf files
    logger.info("Generating conformers and their energies using RDKit")
    # Use canonical SMILES
    smiles = StandardizeSmiles(smiles)
    mol, conformer_energies = generate_conformer_energies(
        smiles=smiles,
        n_conformers=config[mol_type]["n_conformers"],
        args=args,
        config=config,
        logger=logger,
    )
    logger.info("Finished generating conformers")

    # Write conformers to .sdf file
    sdf_file = os.path.join(pre_dir, "conformers.sdf")
    logger.debug(f"Writing conformers and their energies to {sdf_file}")

    with open(sdf_file, "w") as fstream:
        writer = SDWriter(fstream)
        for cid, energy in conformer_energies:
            mol.SetIntProp("conformer_id", cid)
            mol.SetDoubleProp("energy", energy)
            writer.write(mol, confId=cid)
        writer.close()
    logger.debug("Finished writing conformers and their energies")

    # Write conformers to .xyz files
    logger.debug(f"Getting conformer coordinates from {sdf_file}")
    sdfstream = open(sdf_file, "rb")
    mol_supplier = ForwardSDMolSupplier(
        fileobj=sdfstream, sanitize=False, removeHs=False
    )

    # Iterating over conformers
    for conformer in mol_supplier:
        # Conformer id and location of xyz file
        cid = conformer.GetIntProp("conformer_id")
        xyz_file = os.path.join(pre_dir, f"conformer_{cid}.xyz")

        logger.debug(f"Writing conformer {cid} coordinates to {xyz_file}")
        MolToXYZFile(
            mol=mol,
            filename=xyz_file,
            precision=config["rdkit"]["precision"],
        )

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

        logger.debug(f"Running xtb thermodynamic computations on conformer {cid}")
        with open(xtb_output_file, "w") as fstream:
            subprocess.Popen(
                args=xtb_args,
                stdout=fstream,
                stderr=subprocess.STDOUT,
                cwd=conformer_dir,
            ).wait()
        logger.info(f"xTB results saved to {xtb_output_file}")
    sdfstream.close()

    logger.info("Compiling results of xTB computations")
    results = dict(conformer_id=[], enthalpy=[], entropy=[], free_energy=[])

    with os.scandir(post_dir) as folder:
        conformer_dirs = [d for d in folder if d.is_dir()]
    logger.debug(f"Searching conformer dirs {conformer_dirs}")

    for conformer_dir in conformer_dirs:
        xtb_output_file = os.path.join(conformer_dir, f"{conformer_dir.name}.txt")
        logger.debug(f"Searching for results in file {xtb_output_file}")

        cid = conformer_dir.name.split("_")[1]
        results["conformer_id"].append(cid)

        enthalpy, entropy, free_energy = parse_xtb(xtb_output_file, logger)
        logger.debug(
            f"Found enthalpy: {enthalpy}, entropy: {entropy}, free_energy: {free_energy}"
        )
        results["enthalpy"].append(enthalpy)
        results["entropy"].append(entropy)
        results["free_energy"].append(free_energy)

    results_file = os.path.join(post_dir, "results.csv")
    logger.info(f"Writing {mol_type} results to {results_file}")

    results = pd.DataFrame(results)
    results.to_csv(results_file, header=True, index=False)

    logger.info(f"Finished writing {mol_type} results")


def run_xtb(monomer_smiles, n_conformers, args, config, logger):
    """
    Calculates the thermodynamic parameters of the Ring-Opening Reaction (ROR) of a given monomer SMILES
    """
    try:
        run_xtb_conformers(
            smiles=config["initiator"]["smiles"],
            mol_type="initiator",
            args=args,
            config=config,
            logger=logger,
        )

        run_xtb_conformers(
            smiles=monomer_smiles,
            n_conformers=config["initiator"]["n_conformers"],
            mol_type="initiator",
            args=args,
            config=config,
            logger=logger,
        )

        monomer_smiles = StandardizeSmiles(monomer_smiles)

        run_name = f"{count}_{lactone_smiles}_{num_conf}conformers_{num_repeat}repeatunits_{init}initiator_{solvent}solvent_{temp}K"

        repeating_bb = lactone_to_repeating_unit(lactone_smiles)
        num_repeat_units = repeating(num_repeat, "B")
        orientation_str = f"0, {repeating(num_repeat, ' 1, ')} {float(0.5)}"
        num_repeat_orient = tuple(map(float, orientation_str.split(", ")))

        polymer = build_polymer(
            initial_bb, repeating_bb, final_bb, num_repeat_units, num_repeat_orient
        )
        molecule_smiles = stk_to_smiles(polymer)
        ind_poly = 0
        ind_poly = increment(ind_poly)
        molecule = molecule_smiles

        Polylactone(
            molecule,
            ind_poly,
            num_conf,
            solvent,
            temp,
            num_repeat,
            gfn_method,
            threads,
            ohess_level,
            mmffv,
        )
        shutil.move(r"starting_polymers", r"Polymer_1")

        initiator_enthalpy, initiator_entropy, initiator_gibbs = data_extraction(
            "initiator_0/G_all_0.txt",
            "initiator_0/H_all_0.txt",
            "initiator_0/S_all_0.txt",
            energy_option,
            conf_counter,
            temp,
            "initiator",
        )
        lactone_enthalpy, lactone_entropy, lactone_gibbs = data_extraction(
            "Lactone_Ring_0/G_all_0.txt",
            "Lactone_Ring_0/H_all_0.txt",
            "Lactone_Ring_0/S_all_0.txt",
            energy_option,
            conf_counter,
            temp,
            "lactone",
        )
        polymer_enthalpy, polymer_entropy, polymer_gibbs = data_extraction(
            "Polymer_1/G_all_1.txt",
            "Polymer_1/H_all_1.txt",
            "Polymer_1/S_all_1.txt",
            energy_option,
            conf_counter,
            temp,
            "polymer",
        )

        x = run_name
        os.path.join(x)

        if not os.path.exists(x):
            os.makedirs(x)
        else:
            print("File name already exists")

        path = run_name
        shutil.move(r"Lactone_Ring_0", path)
        shutil.move(r"initiator_0", path)
        shutil.move(r"Polymer_1", path)
        shutil.move(r"initial_polymer.xyz", path)

        reactant_enthalpy = initiator_enthalpy + num_repeat * lactone_enthalpy
        reactant_entropy = initiator_entropy + num_repeat * lactone_entropy
        reactant_gibbs = initiator_gibbs + num_repeat * lactone_gibbs

        reaction_enthalpy = float(polymer_enthalpy - reactant_enthalpy)
        reaction_entropy = float(polymer_entropy - reactant_entropy)
        reaction_gibbs = float(polymer_gibbs - reactant_gibbs)

        result_data = {
            "Lactone_SMILES": [lactone_smiles],
            "Repeating_Unit_SMILES": [repeating_bb],
            "Polymer_SMILES": [molecule_smiles],
            "ΔG (J/mol)": [reaction_gibbs],
            "ΔH (J/mol)": [reaction_enthalpy],
            "ΔS (J/mol/K)": [reaction_entropy],
        }

        # Convert to DataFrame
        result_df = pd.DataFrame(result_data)

        # Define the path for the individual CSV file
        individual_csv_path = os.path.join(run_name, f"result_smiles{count}.csv")

        # Save to CSV
        result_df.to_csv(individual_csv_path, index=False)

        # Copy the corresponding conf_pop files to their respective directories
        conf_pop_src_initiator = os.path.join(
            "conf_pop_files", f"initiator_confpop_smiles{conf_counter}.csv"
        )
        conf_pop_dst_initiator = os.path.join(
            run_name, "initiator_0", f"initiator_confpop_smiles{count}.csv"
        )
        shutil.copy(conf_pop_src_initiator, conf_pop_dst_initiator)

        conf_pop_src_lactone = os.path.join(
            "conf_pop_files", f"lactone_confpop_smiles{conf_counter}.csv"
        )
        conf_pop_dst_lactone = os.path.join(
            run_name, "Lactone_Ring_0", f"lactone_confpop_smiles{count}.csv"
        )
        shutil.copy(conf_pop_src_lactone, conf_pop_dst_lactone)

        conf_pop_src_polymer = os.path.join(
            "conf_pop_files", f"polymer_confpop_smiles{conf_counter}.csv"
        )
        conf_pop_dst_polymer = os.path.join(
            run_name, "Polymer_1", f"polymer_confpop_smiles{count}.csv"
        )
        shutil.copy(conf_pop_src_polymer, conf_pop_dst_polymer)

        return (
            molecule_smiles,
            repeating_bb,
            run_name,
            reaction_enthalpy,
            reaction_entropy,
            reaction_gibbs,
        )
    except Exception as e:
        logger.error(f"Error in run_xtb for {lactone_smiles}: {e}")
        raise


def main(args, config, logger):
    # get input monomer SMILES strings

    monomer_smiles = parse_monomer_smiles(args, logger)
    logger.info("Finished reading input monomer smiles")

    # setup output dir
    output_dir = os.path.join(args.output, "step1")
    create_dir(output_dir, logger)

    # run xtb for initiator in ROR reaction
    if config["reaction"]["type"] == "ROR":
        run_xtb_conformers(
            smiles=config["initiator"]["smiles"],
            mol_type="initiator",
            dir_name=config["initiator"]["smiles"],
            args=args,
            config=config,
            logger=logger,
        )
    # iterate over monomers in input list
    for monomer_smiles in monomer_smiles:
        # run xtb for monomer
        run_xtb_conformers(
            smiles=monomer_smiles,
            mol_type="monomer",
            dir_name=monomer_smiles,
            args=args,
            config=config,
            logger=logger,
        )

        # iterate over polymer lengths
        for length in config["reaction"]["lengths"]:
            # build polymer
            polymer_smiles = build_polymer(
                monomer_smiles=monomer_smiles,
                length=length,
                reaction_type=config["reaction"]["type"],
                config=config,
                logger=logger,
            )
            if polymer_smiles is None:
                logger.warning(
                    f"Failed to build polymer of length {length}, moving onto next length"
                )
                continue
            # run xtb for polymer
            run_xtb_conformers(
                smiles=polymer_smiles,
                mol_type="polymer",
                dir_name=f"{monomer_smiles}_{length}",
                args=args,
                config=config,
                logger=logger,
            )

    # run xTB calculations
    count = 1  # Start count from 1
    conf_counter = 1  # Initialize the counter for conf_pop files

    # setup conformer dir
    conformer_dir = os.path.join(args.output, "step1", "conformers")

    logger.info(f"Creating output directory for conformers")
    logger.info(f"Conformer dir: {conformer_dir}")

    try:
        os.mkdir(conformer_dir)
        logger.info(f"Output directory {conformer_dir} created")
    except FileExistsError:
        logger.warning(f"Output directory {conformer_dir} already exists")
        os.makedirs(conformer_dir, exist_ok=True)

    # Initialize dictionary to store the results
    results = {
        "monomer_smiles": [],
        "repeating_block_smiles": [],
        "polymer_smiles": [],
        "Ring Size": [],
        "reaction_enthalpy (J/mol)": [],
        "reaction_entropy (J/mol/K)": [],
        "reaction_gibbs (J/mol)": [],
    }

    for i, smiles in enumerate(monomer_smiles):
        logger.info(f"Processing SMILES ({i + 1}/{len(monomer_smiles)}): {smiles}")

        ring_size = get_ring_size(smiles)
        logger.debug(f"Computed ring size: {ring_size}")

        n_conformers = ring_size * config["num_conf"]
        logger.debug(f"Number of conformers to generate: {n_conformers}")

        try:
            (
                molecule_smiles,
                repeating_bb,
                run_name,
                reaction_enthalpy,
                reaction_entropy,
                reaction_gibbs,
            ) = run_xtb(
                i,
                smiles,
                n_conformers,
                num_repeat,
                initiator_opt,
                solvent,
                temp,
                energy_option,
                conf_counter,
                gfn_method,
                threads,
                ohess_level,
                mmffv,
            )
            if "OOC" not in molecule_smiles:  # WHY?
                # Append results to the lists
                results["count"].append(count)
                results["lactone_smiles"].append(i)
                results["repeating_block_smiles"].append(repeating_bb)
                results["polymer_smiles"].append(molecule_smiles)
                try:
                    ring_size = get_ring_size(i)
                except ValueError as e:
                    logger.error(f"Error getting ring size: {e}")
                    ring_size = "Unknown"
                results["Ring Size"].append(ring_size)
                results["reaction_enthalpy (J/mol)"].append(f"{reaction_enthalpy:.3f}")
                results["reaction_entropy (J/mol/K)"].append(f"{reaction_entropy:.3f}")
                results["reaction_gibbs (J/mol)"].append(f"{reaction_gibbs:.3f}")

                # Increment the counter for each conf_pop file generated
                conf_counter += 1
                count += 1

                # Call process_lactone_entry to save the most stable xtbopt files in the main SMILES folder
                process_lactone_entry(run_name, run_name, run_name)
            else:
                logger.info(
                    f"This polymer {molecule_smiles} contained OOC and was skipped."
                )
                continue
        except Exception as e:
            logger.error(f"run_xtb did not work for SMILES {i}: {e}")
            paths = ["initiator_0", "Lactone_Ring_0", "Polymer_1"]
            for path in paths:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    logger.error(f"The directory {path} does not exist")
            continue

    # Convert results to a DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv("combined_results.csv", index=False)

    print(monomer_smiles)


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser = XTBParser()
    args = parser.parse_args()

    # load logging config from config dir
    try:
        with open(f"{args.config}/logging.toml", "rb") as fstream:
            log_config = tomllib.load(fstream)
            log_config["handlers"]["file"]["filename"] = f"{args.output}/larest.log"
    except Exception as e:
        print(e)
        raise SystemExit(1)

    logging.config.dictConfig(log_config)
    logger = logging.getLogger("Step 1")
    logger.info("Logging config loaded")

    # load step 1 config
    try:
        with open(f"{args.config}/step1.toml", "rb") as fstream:
            config = tomllib.load(fstream)
            logger.info("Step 1 config loaded")
    except Exception as e:
        logger.exception(e)
        logger.error("Failed to load Step 1 config")
        raise SystemExit(1)

    # TODO: write assertions for config

    main(args, config, logger)
