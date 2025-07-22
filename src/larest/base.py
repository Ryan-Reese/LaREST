import argparse
import json
import logging
import subprocess
from abc import ABCMeta
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
    MolToXYZFile,
    SDWriter,
)
from rdkit.Chem.rdmolops import AddHs
from rdkit.ForceField.rdForceField import MMFFMolProperties

# from larest.calculators import run_rdkit
from larest.chem import build_polymer, get_mol, get_ring_size
from larest.constants import (
    CENSO_OUTPUT_PARAMS,
    CENSO_SECTIONS,
    CREST_OUTPUT_PARAMS,
    KCALMOL_TO_JMOL,
    PIPELINE_SECTIONS,
    XTB_OUTPUT_PARAMS,
)
from larest.output import create_dir, slugify
from larest.parsers import (
    extract_best_conformer_xyz,
    parse_best_censo_conformers,
    parse_best_rdkit_conformer,
    parse_censo_output,
    parse_command_args,
    parse_crest_entropy_output,
    parse_xtb_output,
)
from larest.setup import create_censorc


class LarestMol(metaclass=ABCMeta):
    _smiles: str
    _args: argparse.Namespace
    _config: dict[str, Any]
    _logger: logging.Logger
    _xtb_results: dict[str, dict[str, float | None]]
    _entropy_results: dict[str, float | None]
    _censo_results: dict[str, dict[str, float | None]]

    def __init__(
        self,
        smiles: str,
        args: argparse.Namespace,
        config: dict[str, Any],
    ) -> None:
        self.smiles = smiles
        self.args = args
        self.config = config
        self.logger = logging.getLogger(name=self.__class__.__name__)
        self._xtb_results = dict.fromkeys(
            PIPELINE_SECTIONS,
            dict.fromkeys(XTB_OUTPUT_PARAMS, None),
        )
        self._entropy_results = dict.fromkeys(CREST_OUTPUT_PARAMS, None)
        self._censo_results = dict.fromkeys(
            CENSO_SECTIONS,
            dict.fromkeys(CENSO_OUTPUT_PARAMS, None),
        )

    @property
    def smiles(self) -> str:
        return self._smiles

    @property
    def dir_path(self) -> Path:
        return Path(self.args.output) / self.__class__.__name__ / slugify(self.smiles)

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

    @property
    def entropy_results(self) -> dict[str, Any]:
        return self._entropy_results

    @property
    def censo_results(self) -> dict[str, Any]:
        return self._censo_results

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
    def xtb_results(self, xtb_results: dict[str, dict[str, float | None]]) -> None:
        self._xtb_results = xtb_results

    @entropy_results.setter
    def entropy_results(self, xtb_results: dict[str, float | None]) -> None:
        self._entropy_results = xtb_results

    @censo_results.setter
    def censo_results(self, censo_results: dict[str, dict[str, float | None]]) -> None:
        self._censo_results = censo_results

    def run(self) -> None:
        try:
            if self.config["steps"]["rdkit"]:
                self._run_rdkit()
            if self.config["steps"]["crest_confgen"]:
                self._run_crest_confgen()
            if self.config["steps"]["censo"]:
                self._run_censo()
            if self.config["steps"]["crest_entropy"]:
                self._run_crest_entropy()

            self._write_results()
        except Exception:
            self.logger.exception("Error encountered within pipeline, exiting...")
            raise

    def _run_rdkit(self) -> None:
        # setup RDKit dir if not present
        rdkit_dir: Path = self.dir_path / "rdkit"
        create_dir(rdkit_dir, self.logger)

        self.logger.debug(
            "Generating conformers and computing energies using RDKit",
        )
        try:
            mol: Mol = AddHs(
                get_mol(
                    StandardizeSmiles(self.smiles),
                    self.logger,
                ),
            )
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

        # Running xTB for all conformers
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
            xtb_dir: Path = (
                self.dir_path / "xtb" / "rdkit" / f"conformer_{conformer_id}"
            )
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
        xtb_results |= {param: [] for param in XTB_OUTPUT_PARAMS}

        xtb_dir: Path = self.dir_path / "xtb" / "rdkit"
        conformer_dirs: list[Path] = [d for d in xtb_dir.iterdir() if d.is_dir()]
        self.logger.debug(f"Searching conformer dirs {conformer_dirs}")

        for conformer_dir in conformer_dirs:
            xtb_output_file: Path = conformer_dir / f"{conformer_dir.name}.txt"

            try:
                xtb_output = parse_xtb_output(
                    xtb_output_file=xtb_output_file,
                    temperature=self.config["xtb"]["rdkit"]["etemp"],
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
                for param in XTB_OUTPUT_PARAMS:
                    xtb_results[param].append(xtb_output[param])

        xtb_results_file: Path = xtb_dir / "results.csv"
        self.logger.debug(f"Writing results to {xtb_results_file}")

        xtb_results_df = pd.DataFrame(xtb_results, dtype=np.float64).sort_values(
            "G",
        )
        xtb_results_df.to_csv(xtb_results_file, header=True, index=False)

        self.logger.debug(
            f"Finished writing results for {self.__class__.__name__} ({self.smiles})",
        )

        # add to self.results
        best_rdkit_conformer_results = xtb_results_df.iloc[0].to_dict()
        del best_rdkit_conformer_results["conformer_id"]
        self.xtb_results = self.xtb_results | {"rdkit": best_rdkit_conformer_results}

    def _run_crest_confgen(self) -> None:
        """
        Running the CREST standard procedure to generate a conformer/rotamer ensemble.
        Subsequently performing thermo calculations using xTB on best conformer (if desired).
        """
        crest_dir: Path = self.dir_path / "crest_confgen"
        create_dir(crest_dir, self.logger)

        xtb_rdkit_dir: Path = self.dir_path / "xtb" / "rdkit"
        best_rdkit_conformer_id: int = int(
            parse_best_rdkit_conformer(xtb_rdkit_dir)["conformer_id"],
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
            xtb_dir: Path = self.dir_path / "xtb" / "crest"
            create_dir(xtb_dir, self.logger)
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

        # create .censo2rc for run
        create_censorc(args=self.args, logger=self.logger)

        censo_dir: Path = self.dir_path / "censo"
        create_dir(censo_dir, self.logger)

        # specify location for censo log file
        censo_output_file: Path = censo_dir / "censo.txt"
        censo_config_file: Path = Path(self.args.config) / "temp" / ".censo2rc"

        # specify location for crest conformers file
        crest_conformers_file: Path = (
            self.dir_path / "crest_confgen" / "crest_conformers.xyz"
        )

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

        censo_results: dict[str, dict[str, float | None]] = parse_censo_output(
            censo_output_file=censo_output_file,
            temperature=self.config["censo"]["general"]["temperature"],
            logger=self.logger,
        )

        self.censo_results |= censo_results

        if self.config["steps"]["xtb"]:
            best_censo_conformers: dict[str, str] = parse_best_censo_conformers(
                censo_output_file=censo_output_file,
                logger=self.logger,
            )
            for section in CENSO_SECTIONS:
                if best_censo_conformers[section] is None:
                    continue

                censo_conformers_xyz_file: Path = censo_dir / f"{section}.xyz"

                xtb_dir: Path = self.dir_path / "xtb" / "censo" / section
                create_dir(xtb_dir, self.logger)

                best_conformer_xyz_file: Path = xtb_dir / f"{section}.xyz"

                extract_best_conformer_xyz(
                    censo_conformers_xyz_file=censo_conformers_xyz_file,
                    best_conformer_id=best_censo_conformers[section],
                    output_xyz_file=best_conformer_xyz_file,
                    logger=self.logger,
                )

                xtb_results: dict[str, float | None] = self._run_xtb(
                    xtb_input_file=best_conformer_xyz_file,
                    xtb_dir=xtb_dir,
                    xtb_sub_config=["xtb", "censo"],
                )

                # add to self.results
                self.xtb_results = self.xtb_results | {section: xtb_results}

    def _run_xtb(
        self,
        xtb_input_file: Path,
        xtb_dir: Path,
        xtb_sub_config: list[str],
    ) -> dict[str, float | None]:
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
            f"Finished writing results for {self.__class__.__name__}",
        )

        return xtb_results

    def _write_results(self) -> None:
        xtb_results_file: Path = self.dir_path / "xtb" / "results.json"

        with open(xtb_results_file, "w") as fstream:
            json.dump(
                self.xtb_results,
                fstream,
                sort_keys=True,
                indent=4,
                allow_nan=True,
            )

        censo_results_file: Path = self.dir_path / "censo" / "results.json"

        with open(censo_results_file, "w") as fstream:
            json.dump(
                self.censo_results,
                fstream,
                sort_keys=True,
                indent=4,
                allow_nan=True,
            )

    def _run_crest_entropy(self) -> None:
        """
        Running CREST to estimate the conformational entropy of the ensemble.
        """
        crest_dir: Path = self.dir_path / "crest_entropy"
        create_dir(crest_dir, self.logger)

        best_censo_conformer_xyz_file: Path = (
            self.dir_path / "censo" / "3_REFINEMENT.xyz"
        )

        # specify location for crest log file
        crest_output_file: Path = crest_dir / "crest.txt"

        # conformer generation with CREST
        crest_args: list[str] = [
            "crest",
            str(best_censo_conformer_xyz_file.absolute()),
        ]
        crest_args += parse_command_args(
            sub_config=["crest", "entropy"],
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

        crest_results: dict[str, float | None] = parse_crest_entropy_output(
            crest_output_file=crest_output_file,
            logger=self.logger,
        )

        self.entropy_results = self.entropy_results | crest_results

        if self.config["steps"]["xtb"]:
            crest_corrected_results: dict[str, float | None] = self.xtb_results[
                "crest"
            ].copy()
            crest_corrected_results["S"] += self.entropy_results["S_total"]
            self.xtb_results["crest_corrected"] = crest_corrected_results

        if self.config["steps"]["censo"]:
            censo_corrected_results: dict[str, float | None] = self.censo_results[
                "3_REFINEMENT"
            ].copy()
            censo_corrected_results["S"] += self.entropy_results["S_total"]

            self.censo_results["censo_corrected"] = censo_corrected_results

        crest_results_file: Path = crest_dir / "results.json"
        self.logger.debug(f"Writing results to {crest_results_file}")

        with open(crest_results_file, "w") as fstream:
            json.dump(
                crest_results,
                fstream,
                sort_keys=True,
                allow_nan=True,
                indent=4,
            )

        self.logger.debug(
            f"Finished writing results for {self.__class__.__name__} ({self.smiles})",
        )


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

    @LarestMol.dir_path.getter
    def dir_path(self) -> Path:
        return (
            Path(self.args.output)
            / self.__class__.__name__
            / f"{slugify(self._smiles)}_{self.length}"
        )

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
    _initiator: Initiator
    _polymers: list[Polymer]

    @property
    def ring_size(self) -> int | None:
        return self._ring_size

    @property
    def initiator(self) -> Initiator:
        return self._initiator

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

        self._initiator = Initiator(
            smiles=config["reaction"]["initiator"],
            args=self.args,
            config=self.config,
        )
        self._polymers = [
            Polymer(
                smiles=self.smiles,
                length=length,
                args=self.args,
                config=self.config,
            )
            for length in self.config["reaction"]["lengths"]
        ]

    def compile_results(self) -> None:
        summary_dir: Path = self.dir_path / "summary"
        create_dir(summary_dir, self.logger)

        # iterating over pipeline sections for xTB
        if self.config["steps"]["xtb"]:
            for section in self.xtb_results:
                xtb_summary_headings: list[str] = ["polymer_length"]
                for mol_type in ["monomer", "initiator", "polymer"]:
                    xtb_summary_headings.extend(
                        [f"{mol_type}_{param}" for param in XTB_OUTPUT_PARAMS],
                    )

                xtb_summary: dict[str, list[float]] = {
                    heading: [] for heading in xtb_summary_headings
                }

                # iterating over polymer lengths
                for polymer in self.polymers:
                    # adding monomer and polymer results
                    xtb_summary["polymer_length"].append(polymer.length)
                    for param in XTB_OUTPUT_PARAMS:
                        xtb_summary[f"polymer_{param}"].append(
                            polymer.xtb_results[section][param],
                        )
                        xtb_summary[f"monomer_{param}"].append(
                            self.xtb_results[section][param],
                        )
                        xtb_summary[f"initiator_{param}"].append(
                            self.initiator.xtb_results[section][param]
                            if (self.config["reaction"]["type"] == "ROR")
                            else 0,
                        )

                # converting to df
                xtb_summary_df: pd.DataFrame = pd.DataFrame(
                    xtb_summary,
                    index=None,
                    dtype=np.float64,
                ).sort_values(
                    "polymer_length",
                    ascending=True,
                )

                # calculating deltas
                for param in XTB_OUTPUT_PARAMS:
                    xtb_summary_df[f"delta_{param}"] = (
                        xtb_summary_df[f"polymer_{param}"]
                        - (
                            xtb_summary_df["polymer_length"]
                            * xtb_summary_df[f"monomer_{param}"]
                        )
                        - xtb_summary_df[f"initiator_{param}"]
                    ) / xtb_summary_df["polymer_length"]

                # specify to save results
                xtb_summary_file: Path = summary_dir / f"{section}_xtb.csv"

                xtb_summary_df.to_csv(xtb_summary_file, header=True, index=False)

        # iterating over pipeline sections for CENSO
        if self.config["steps"]["censo"]:
            for section in self.censo_results:
                censo_summary_headings: list[str] = ["polymer_length"]
                for mol_type in ["monomer", "initiator", "polymer"]:
                    censo_summary_headings.extend(
                        [f"{mol_type}_{param}" for param in CENSO_OUTPUT_PARAMS],
                    )

                censo_summary: dict[str, list[float]] = {
                    heading: [] for heading in censo_summary_headings
                }

                # iterating over polymer lengths
                for polymer in self.polymers:
                    # adding monomer and polymer results
                    censo_summary["polymer_length"].append(polymer.length)
                    for param in CENSO_OUTPUT_PARAMS:
                        censo_summary[f"polymer_{param}"].append(
                            polymer.censo_results[section][param],
                        )
                        censo_summary[f"monomer_{param}"].append(
                            self.censo_results[section][param],
                        )
                        censo_summary[f"initiator_{param}"].append(
                            self.initiator.censo_results[section][param]
                            if (self.config["reaction"]["type"] == "ROR")
                            else 0,
                        )

                # converting to df
                censo_summary_df: pd.DataFrame = pd.DataFrame(
                    censo_summary,
                    index=None,
                    dtype=np.float64,
                ).sort_values(
                    "polymer_length",
                    ascending=True,
                )

                # calculating deltas
                for param in CENSO_OUTPUT_PARAMS:
                    censo_summary_df[f"delta_{param}"] = (
                        censo_summary_df[f"polymer_{param}"]
                        - (
                            censo_summary_df["polymer_length"]
                            * censo_summary_df[f"monomer_{param}"]
                        )
                        - censo_summary_df[f"initiator_{param}"]
                    ) / censo_summary_df["polymer_length"]

                # specify to save results
                censo_summary_file: Path = summary_dir / f"{section}.csv"

                censo_summary_df.to_csv(censo_summary_file, header=True, index=False)
