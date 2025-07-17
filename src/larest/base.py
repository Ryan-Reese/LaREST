import argparse
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
from larest.constants import KCALMOL_TO_JMOL, XTB_OUTPUT_HEADINGS
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

    def run(self) -> None:
        # self._run_rdkit()
        # self._run_crest()
        self._run_censo()

    def _run_rdkit(self) -> None:
        # setup RDKit dir if not present
        mol_dir: Path = Path(self.args.output) / self.__class__.__name__ / self.dirname
        rdkit_dir: Path = mol_dir / "rdkit"
        create_dir(rdkit_dir, self.logger)

        self.logger.debug(
            "Generating conformers and computing energies using RDKit",
        )
        mol: Mol = AddHs(get_mol(self.smiles, self.logger))
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
        for conformer in tqdm(
            mol_supplier,
            desc="iterating over conformers",
            total=n_conformers,
        ):
            # Conformer id and location of xyz file
            cid: int = conformer.GetIntProp("conformer_id")
            xyz_file: Path = rdkit_dir / f"conformer_{cid}.xyz"

            try:
                MolToXYZFile(
                    mol=mol,
                    filename=xyz_file,
                    precision=self.config["rdkit"]["precision"],
                )
            except Exception as err:
                self.logger.exception(err)
                self.logger.exception(
                    f"Failed to write conformer coordinates to {xyz_file}",
                )
                raise

            # Creating output dir for xTB thermo calculation
            xtb_dir: Path = mol_dir / "xtb" / "rdkit" / f"conformer_{cid}"
            create_dir(xtb_dir, self.logger)

            # Specify location for xtb log file
            xtb_output_file: Path = xtb_dir / f"conformer_{cid}.txt"

            # Optimisation with xTB
            xtb_args: list[str] = [
                "xtb",
                f"../../../rdkit/conformer_{cid}.xyz",
                # implement above using path
                "--namespace",
                f"conformer_{cid}",
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

    def _run_crest(self) -> None:
        """
        Running the CREST standard procedure to generate a conformer/rotamer ensemble.
        Subsequently performing thermo calculations using xTB on best conformer.
        """
        mol_dir: Path = Path(self.args.output) / self.__class__.__name__ / self.dirname
        crest_dir: Path = mol_dir / "crest"
        create_dir(crest_dir, self.logger)

        # Copy most stable conformer from step 1
        xtb_rdkit_dir: Path = mol_dir / "xtb" / "rdkit"
        best_rdkit_conformer_id: int = int(
            parse_best_rdkit_conformer(xtb_rdkit_dir)["conformer_id"]
        )

        # specify location for crest log file
        crest_output_file: Path = crest_dir / "crest.txt"

        # conformer generation with CREST
        crest_args: list[str] = [
            "crest",
            f"../xtb/rdkit/conformer_{best_rdkit_conformer_id}/conformer_{best_rdkit_conformer_id}.xtbopt.xyz",
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
            self.logger.exception(
                f"Failed to run CREST command with arguments {crest_args}",
            )
            raise

        # best_crest_conformer_xyz_file: Path = crest_dir / "crest_best.xyz"
        xtb_dir: Path = mol_dir / "xtb" / "crest"
        create_dir(xtb_dir, self.logger)

        xtb_output_file: Path = xtb_dir / "xtb.txt"

        # Optimisation with xTB
        xtb_args: list[str] = [
            "xtb",
            "../../crest/crest_best.xyz",
            # implement above using path
            "--namespace",
            "crest_best",
        ]
        xtb_args += parse_command_args(
            sub_config=["xtb", "crest"],
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
            raise

        xtb_results: dict[str, list[float | None]] = {
            heading: [] for heading in XTB_OUTPUT_HEADINGS
        }

        try:
            xtb_output = parse_xtb_output(
                xtb_output_file=xtb_output_file,
                temperature=self.config["xtb"]["crest"]["etemp"],
                config=self.config,
                logger=self.logger,
            )
        except Exception:
            self.logger.exception(
                f"Failed to parse xtb results for crest_best conformer",
            )
        else:
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

    def _run_censo(self) -> None:
        create_censorc(args=self.args, logger=self.logger)


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

    @property
    def polymer_smiles(self) -> str:
        return build_polymer(
            monomer_smiles=self.smiles,
            polymer_length=self.length,
            reaction_type=self.config["reaction"]["type"],
            config=self.config,
            logger=self.logger,
        )

    @property
    def dirname(self) -> str:
        return f"{self.smiles}_{self.length}"

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
                smiles=self.smiles, length=length, args=self.args, config=self.config
            )
            for length in self.config["reaction"]["lengths"]
        ]

    def _compile_monomer_results(self):
        pass
