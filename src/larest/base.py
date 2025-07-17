import argparse
import json
import logging
import subprocess
from abc import ABCMeta, abstractmethod
from io import IOBase
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit.Chem.AllChem import (
    MMFFGetMoleculeForceField,
    MMFFGetMoleculeProperties,
)
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
from rdkit.ForceField.rdForceField import MMFFMolProperties
from tqdm import tqdm

# from larest.calculators import run_rdkit
from larest.chem import build_polymer, get_mol, get_ring_size
from larest.constants import CENSO_HEADINGS, KCALMOL_TO_JMOL, XTB_OUTPUT_HEADINGS
from larest.output import create_dir, slugify
from larest.parsers import (
    parse_best_rdkit_conformer,
    parse_command_args,
    parse_xtb_output,
)
from larest.setup import create_censorc, get_logger


class LarestMol(metaclass=ABCMeta):
    _smiles: str
    _args: argparse.Namespace
    _config: dict[str, Any]
    _logger: logging.Logger
    _xtb_results: dict[str, Any]

    def __init__(
        self,
        smiles: str,
        args: argparse.Namespace,
        config: dict[str, Any],
    ) -> None:
        self.smiles = smiles
        self.args = args
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            args=self.args,
            config=self.config,
        )
        self._xtb_results = {}

    @property
    def smiles(self) -> str:
        return self._smiles

    @property
    def dirname(self) -> str:
        return slugify(self.smiles)

    @property
    def args(self) -> argparse.Namespace:
        return self._args

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def xtb_results(self) -> dict[str, Any]:
        return self._xtb_results

    @smiles.setter
    def smiles(self, smiles: str) -> None:
        self._smiles = smiles

    @args.setter
    def args(self, args: argparse.Namespace) -> None:
        self._args = args

    @config.setter
    def config(self, config: dict[str, Any]) -> None:
        self._config = config

    @logger.setter
    def logger(self, logger: logging.Logger) -> None:
        self._logger = logger

    @xtb_results.setter
    def xtb_results(self, xtb_results: dict[str, Any]) -> None:
        self._xtb_results = xtb_results

    def run(self) -> None:
        try:
            if self.config["steps"]["rdkit"]:
                self._run_rdkit()
            if self.config["steps"]["crest"]:
                self._run_crest()
            if self.config["steps"]["censo"]:
                create_censorc(args=self.args, logger=self.logger)
                self._run_censo()
            if self.config["steps"]["xtb"]:
                self._write_xtb_results()
        except Exception:
            self.logger.exception("Error encountered within pipeline, exiting...")
            raise

    def _write_xtb_results(self) -> None:
        mol_dir: Path = Path(self.args.output) / self.__class__.__name__ / self.dirname
        xtb_results_file: Path = mol_dir / "xtb" / "results.json"

        with open(xtb_results_file, "w") as fstream:
            json.dump(
                self.xtb_results,
                fstream,
                sort_keys=True,
                indent=4,
                allow_nan=True,
            )

    def _run_rdkit(self) -> None:
        # setup RDKit dir if not present
        mol_dir: Path = Path(self.args.output) / self.__class__.__name__ / self.dirname
        rdkit_dir: Path = mol_dir / "rdkit"
        create_dir(rdkit_dir, self.logger)

        self.logger.debug(
            "Generating conformers and computing energies using RDKit",
        )
        try:
            mol: Mol = AddHs(get_mol(self.smiles, self.logger))
        except Exception as err:
            self.logger.exception(err)
            raise

        n_conformers: int = self.config["rdkit"]["n_conformers"]

        self.logger.debug(
            f"Generating {n_conformers} conformers",
        )
        conformer_ids: list[int] = EmbedMultipleConfs(
            mol,
            n_conformers,
            useRandomCoords=True,
            randomSeed=self.config["rdkit"]["random_seed"],
            boxSizeMult=self.config["rdkit"]["conformer_box_size"],
            numThreads=self.config["rdkit"]["n_cores"],
        )

        self.logger.debug(f"Optimising the {n_conformers} conformers")

        MMFFOptimizeMoleculeConfs(
            mol,
            numThreads=self.config["rdkit"]["n_cores"],
            maxIters=self.config["rdkit"]["mmff_iters"],
            mmffVariant=self.config["rdkit"]["mmff"],
        )

        self.logger.debug("Computing molecular properties for MMFF")
        mp: MMFFMolProperties = MMFFGetMoleculeProperties(
            mol,
            mmffVariant=self.config["rdkit"]["mmff"],
            mmffVerbosity=int(self.args.verbose),
        )

        self.logger.debug(f"Computing energies for the {n_conformers} conformers")
        conformer_energies: list[tuple[int, float]] = sorted(
            [
                (
                    conformer_id,
                    MMFFGetMoleculeForceField(mol, mp, confId=conformer_id).CalcEnergy()
                    * KCALMOL_TO_JMOL,
                )
                for conformer_id in conformer_ids
            ],
            key=lambda x: x[1],  # sort by energies
        )

        self.logger.debug(f"Aligning the {n_conformers} conformers by their geometries")
        AlignMolConformers(mol, maxIters=self.config["rdkit"]["align_iters"])

        self.logger.debug("Finished generating conformers")

        sdf_file: Path = rdkit_dir / "conformers.sdf"
        self.logger.debug(f"Writing conformers and their energies to {sdf_file}")

        try:
            with open(sdf_file, "w") as fstream:
                writer: SDWriter = SDWriter(fstream)
                for cid, energy in conformer_energies:
                    mol.SetIntProp("conformer_id", cid)
                    mol.SetDoubleProp("energy", energy)
                    writer.write(mol, confId=cid)
                writer.close()
        except Exception as err:
            self.logger.exception(err)
            self.logger.exception(f"Failed to write RDKit conformers to {sdf_file}")
            raise
        else:
            self.logger.debug("Finished writing conformers and their energies")

        # Write conformers to .xyz files
        self.logger.debug(f"Getting conformer coordinates from {sdf_file}")
        sdfstream: IOBase = open(sdf_file, "rb")
        mol_supplier: ForwardSDMolSupplier = ForwardSDMolSupplier(
            fileobj=sdfstream,
            sanitize=False,
            removeHs=False,
        )

        # Iterating over conformers
        self.logger.debug("Computing thermodynamic parameters of conformers using xTB")
        for conformer in mol_supplier:
            # Conformer id and location of xyz file
            conformer_id: int = conformer.GetIntProp("conformer_id")
            conformer_xyz_file: Path = rdkit_dir / f"conformer_{conformer_id}.xyz"

            try:
                MolToXYZFile(
                    mol=mol,
                    filename=conformer_xyz_file,
                    precision=self.config["rdkit"]["precision"],
                )
            except Exception as err:
                self.logger.exception(err)
                self.logger.exception(
                    f"Failed to write conformer coordinates to {conformer_xyz_file}",
                )
                raise

            # Creating output dir for xTB thermo calculation
            xtb_dir: Path = mol_dir / "xtb" / "rdkit" / f"conformer_{conformer_id}"
            create_dir(xtb_dir, self.logger)

            # Specify location for xtb log file
            xtb_output_file: Path = xtb_dir / f"conformer_{conformer_id}.txt"

            # Optimisation with xTB
            xtb_args: list[str] = [
                "xtb",
                str(conformer_xyz_file.absolute()),
                "--namespace",
                f"conformer_{conformer_id}",
            ]
            xtb_args += parse_command_args(
                sub_config=["xtb", "rdkit"],
                config=self.config,
                logger=self.logger,
            )

            try:
                with open(xtb_output_file, "w") as fstream:
                    subprocess.Popen(
                        args=xtb_args,
                        stdout=fstream,
                        stderr=subprocess.STDOUT,
                        cwd=xtb_dir,
                    ).wait()
            except Exception as err:
                self.logger.exception(err)
                self.logger.exception(
                    f"Failed to run xTB command with arguments {xtb_args}",
                )
        sdfstream.close()
        self.logger.debug("Finished running xTB on conformers")

        self.logger.debug("Compiling results of xTB computations")

        xtb_results: dict[str, list[float | None]] = {"conformer_id": []}
        xtb_results |= {heading: [] for heading in XTB_OUTPUT_HEADINGS}

        xtb_dir: Path = mol_dir / "xtb" / "rdkit"
        conformer_dirs: list[Path] = [d for d in xtb_dir.iterdir() if d.is_dir()]
        self.logger.debug(f"Searching conformer dirs {conformer_dirs}")

        for conformer_dir in conformer_dirs:
            xtb_output_file: Path = conformer_dir / f"{conformer_dir.name}.txt"

            try:
                xtb_output = parse_xtb_output(
                    xtb_output_file=xtb_output_file,
                    temperature=self.config["xtb"]["rdkit"]["etemp"],
                    config=self.config,
                    logger=self.logger,
                )
            except Exception:
                self.logger.exception(
                    f"Failed to parse xtb results for conformer in {conformer_dir.name}",
                )
                continue
            else:
                xtb_results["conformer_id"].append(
                    int(conformer_dir.name.split("_")[1]),
                )
                for heading in XTB_OUTPUT_HEADINGS:
                    xtb_results[heading].append(xtb_output[heading])

        xtb_results_file: Path = xtb_dir / "results.csv"
        self.logger.debug(f"Writing results to {xtb_results_file}")

        xtb_results_df = pd.DataFrame(xtb_results, dtype=np.float64).sort_values(
            "free_energy",
        )
        xtb_results_df.to_csv(xtb_results_file, header=True, index=False)

        self.logger.debug(
            f"Finished writing results for {self.__class__.__name__} ({self.smiles})",
        )

        # add to self.results
        best_rdkit_conformer_results = xtb_results_df.iloc[0].to_dict()
        del best_rdkit_conformer_results["conformer_id"]
        self.xtb_results = self.xtb_results | {"rdkit": best_rdkit_conformer_results}

    def _run_crest(self) -> None:
        """
        Running the CREST standard procedure to generate a conformer/rotamer ensemble.
        Subsequently performing thermo calculations using xTB on best conformer (if desired).
        """
        mol_dir: Path = Path(self.args.output) / self.__class__.__name__ / self.dirname
        crest_dir: Path = mol_dir / "crest"
        create_dir(crest_dir, self.logger)

        xtb_rdkit_dir: Path = mol_dir / "xtb" / "rdkit"
        best_rdkit_conformer_id: int = int(
            parse_best_rdkit_conformer(xtb_rdkit_dir)["conformer_id"]
        )
        best_rdkit_conformer_xyz_file: Path = (
            xtb_rdkit_dir
            / f"conformer_{best_rdkit_conformer_id}"
            / f"conformer_{best_rdkit_conformer_id}.xtbopt.xyz"
        )

        # specify location for crest log file
        crest_output_file: Path = crest_dir / "crest.txt"

        # conformer generation with CREST
        crest_args: list[str] = [
            "crest",
            str(best_rdkit_conformer_xyz_file.absolute()),
        ]
        crest_args += parse_command_args(
            sub_config=["crest", "confgen"],
            config=self.config,
            logger=self.logger,
        )

        try:
            with open(crest_output_file, "w") as fstream:
                subprocess.Popen(
                    args=crest_args,
                    stdout=fstream,
                    stderr=subprocess.STDOUT,
                    cwd=crest_dir,
                ).wait()
        except Exception as err:
            self.logger.exception(err)
            self.logger.exception(f"Failed to run CREST with arguments {crest_args}")
            raise

        if self.config["steps"]["xtb"]:
            best_crest_conformer_xyz_file: Path = crest_dir / "crest_best.xyz"
            xtb_dir: Path = mol_dir / "xtb" / "crest"
            xtb_output_file: Path = xtb_dir / "xtb.txt"
            xtb_results_file: Path = xtb_dir / "results.csv"
            xtb_results: dict[str, float | None] = self._run_xtb(
                xtb_input_file=best_crest_conformer_xyz_file,
                xtb_dir=xtb_dir,
                xtb_sub_config=["xtb", "crest"],
            )

            # add to self.results
            self.xtb_results = self.xtb_results | {"crest": xtb_results}

    def _run_censo(self) -> None:
        """
        Running CENSO to DFT refine the conformer ensemble from CREST.
        Subsequently performing thermo calculations using xTB on best conformers (if desired)
        """
        mol_dir: Path = Path(self.args.output) / self.__class__.__name__ / self.dirname
        censo_dir: Path = mol_dir / "censo"
        create_dir(censo_dir, self.logger)

        # specify location for censo log file
        censo_output_file: Path = censo_dir / "censo.txt"
        censo_config_file: Path = Path(self.args.config) / "temp" / ".censo2rc"

        # specify location for crest conformers file
        crest_conformers_file: Path = mol_dir / "crest" / "crest_conformers.xyz"

        # conformer ensemble optimisation with CENSO
        censo_args: list[str] = [
            "censo",
            "--input",
            str(crest_conformers_file.absolute()),
            "--inprc",
            str(censo_config_file.absolute()),
        ]
        censo_args += parse_command_args(
            sub_config=["censo", "cli"],
            config=self.config,
            logger=self.logger,
        )
        try:
            with open(censo_output_file, "w") as fstream:
                subprocess.Popen(
                    args=censo_args,
                    stdout=fstream,
                    stderr=subprocess.STDOUT,
                    cwd=censo_dir,
                ).wait()
        except Exception as err:
            self.logger.exception(err)
            self.logger.exception(f"Failed to run CENSO with arguments {censo_args}")
            raise

        if self.config["steps"]["xtb"]:
            for censo_heading in CENSO_HEADINGS:
                best_censo_conformer_xyz_file: Path = censo_dir / f"{censo_heading}.xyz"
                xtb_dir: Path = mol_dir / "xtb" / "censo" / censo_heading
                xtb_results: dict[str, float | None] = self._run_xtb(
                    xtb_input_file=best_censo_conformer_xyz_file,
                    xtb_dir=xtb_dir,
                    xtb_sub_config=["xtb", "censo"],
                )

                # add to self.results
                self.xtb_results = self.xtb_results | {censo_heading: xtb_results}

    def _run_xtb(
        self,
        xtb_input_file: Path,
        xtb_dir: Path,
        xtb_sub_config: list[str],
    ) -> dict[str, float | None]:
        create_dir(xtb_dir, self.logger)

        # Optimisation with xTB
        xtb_args: list[str] = [
            "xtb",
            str(xtb_input_file.absolute()),
            "--namespace",
            xtb_input_file.name.split(".")[0],
        ]
        xtb_args += parse_command_args(
            sub_config=xtb_sub_config,
            config=self.config,
            logger=self.logger,
        )

        xtb_output_file: Path = xtb_dir / "xtb.txt"
        try:
            with open(xtb_output_file, "w") as fstream:
                subprocess.Popen(
                    args=xtb_args,
                    stdout=fstream,
                    stderr=subprocess.STDOUT,
                    cwd=xtb_dir,
                ).wait()
        except Exception as err:
            self.logger.exception(err)
            self.logger.exception(
                f"Failed to run xTB with arguments {xtb_args}",
            )
            raise

        try:
            xtb_results = parse_xtb_output(
                xtb_output_file=xtb_output_file,
                temperature=self.config["xtb"][xtb_sub_config[1]]["etemp"],
                config=self.config,
                logger=self.logger,
            )
        except Exception:
            self.logger.exception(
                f"Failed to parse xtb results in file {xtb_output_file}",
            )
            raise

        xtb_results_file: Path = xtb_dir / "results.json"
        self.logger.debug(f"Writing results to {xtb_results_file}")

        with open(xtb_results_file, "w") as fstream:
            json.dump(
                xtb_results,
                fstream,
                sort_keys=True,
                allow_nan=True,
                indent=4,
            )

        self.logger.debug(
            f"Finished writing results for {self.__class__.__name__} ({self.smiles})",
        )

        return xtb_results


class Initiator(LarestMol):
    def __init__(
        self,
        smiles: str,
        args: argparse.Namespace,
        config: dict[str, Any],
    ) -> None:
        super().__init__(smiles=smiles, args=args, config=config)


class Polymer(LarestMol):
    _length: int
    _polymer_smiles: str

    @property
    def length(self) -> int:
        return self._length

    @LarestMol.dirname.getter
    def dirname(self) -> str:
        return f"{slugify(self._smiles)}_{self.length}"

    @LarestMol.smiles.getter
    def smiles(self) -> str:
        try:
            return build_polymer(
                monomer_smiles=self._smiles,
                polymer_length=self.length,
                reaction_type=self.config["reaction"]["type"],
                config=self.config,
                logger=self.logger,
            )
        except Exception:
            self.logger.exception("Failed to get polymer smiles")
            self.logger.warning("Using monomer smiles instead")
            raise

    @length.setter
    def length(self, length: int) -> None:
        self._length = length

    def __init__(
        self,
        smiles: str,
        length: int,
        args: argparse.Namespace,
        config: dict[str, Any],
    ) -> None:
        super().__init__(smiles=smiles, args=args, config=config)
        self.length = length


class Monomer(LarestMol):
    _ring_size: int | None
    _polymers: list[Polymer]

    @property
    def ring_size(self) -> int | None:
        return self._ring_size

    @property
    def polymers(self) -> list[Polymer]:
        return self._polymers

    @ring_size.setter
    def ring_size(self) -> None:
        self._ring_size = get_ring_size(smiles=self.smiles, logger=self.logger)

    def __init__(
        self,
        smiles: str,
        args: argparse.Namespace,
        config: dict[str, Any],
    ) -> None:
        super().__init__(smiles=smiles, args=args, config=config)
        self._polymers = [
            Polymer(
                smiles=self.smiles,
                length=length,
                args=self.args,
                config=self.config,
            )
            for length in self.config["reaction"]["lengths"]
        ]

    def compile_monomer_results(self) -> None:
        pass
