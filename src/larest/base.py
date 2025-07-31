from __future__ import annotations

import json
import logging
import subprocess
from abc import ABCMeta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from rdkit.Chem.AllChem import (
    MMFFGetMoleculeForceField,
    MMFFGetMoleculeProperties,
)
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMoleculeConfs
from rdkit.Chem.rdMolAlign import AlignMolConformers
from rdkit.Chem.rdmolfiles import (
    ForwardSDMolSupplier,
    MolToXYZFile,
    SDWriter,
)
from rdkit.Chem.rdmolops import AddHs

from larest.checkpoint import PipelineStage, restore_results
from larest.chem import build_polymer, get_mol, get_ring_size
from larest.constants import (
    KCALMOL_TO_JMOL,
    PIPELINE_SECTIONS,
    THERMODYNAMIC_PARAMS,
)
from larest.exceptions import NoResultsError
from larest.output import create_dir, remove_dir, slugify
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

if TYPE_CHECKING:
    from argparse import Namespace
    from io import IOBase

    from rdkit.Chem.rdchem import Mol
    from rdkit.ForceField.rdForceField import MMFFMolProperties


class LarestMol(metaclass=ABCMeta):
    _smiles: str
    _args: Namespace
    _config: dict[str, Any]
    _logger: logging.Logger
    _pipeline_stage: PipelineStage
    _results: dict[str, dict[str, float | None]]

    def __init__(
        self,
        smiles: str,
        args: Namespace,
        config: dict[str, Any],
    ) -> None:
        self._args = args
        self._config = config
        self._logger = logging.getLogger(name=self.__class__.__name__)

        self.smiles = smiles
        self._results = {
            section: dict.fromkeys(
                THERMODYNAMIC_PARAMS,
                None,
            )
            for section in PIPELINE_SECTIONS
        }
        self._results, self._pipeline_stage = restore_results(
            results=self._results,
            dir_path=self.dir_path,
            logger=self._logger,
        )

    @property
    def dir_path(self) -> Path:
        return Path(
            self._args.output,
            self.__class__.__name__,
            slugify(self.smiles),
        )

    @property
    def results(self) -> dict[str, dict[str, float | None]]:
        return self._results

    @property
    def smiles(self) -> str:
        return self._smiles

    @smiles.setter
    def smiles(self, smiles: str) -> None:
        try:
            get_mol(smiles, self._logger)
        except Exception as err:
            self._logger.exception(err)
            raise
        else:
            self._smiles = smiles

    def run(self) -> None:
        """Run the LaREST pipeline for the molecule"""
        try:
            if self._config["steps"]["rdkit"] and (
                self._pipeline_stage <= PipelineStage.RDKIT
            ):
                self._run_rdkit()
            if self._config["steps"]["crest_confgen"] and (
                self._pipeline_stage <= PipelineStage.CREST_CONFGEN
            ):
                self._run_crest_confgen()
            if self._config["steps"]["censo"] and (
                self._pipeline_stage <= PipelineStage.CENSO
            ):
                self._run_censo()
            if self._config["steps"]["crest_entropy"] and (
                self._pipeline_stage <= PipelineStage.CREST_ENTROPY
            ):
                self._run_crest_entropy()

            self._write_final_results()
            self._cleanup_pipeline()
        except Exception:
            self._logger.exception("Error encountered within pipeline, exiting...")
            raise

    def _run_rdkit(self) -> None:
        # setup RDKit dir if not present
        rdkit_dir: Path = Path(self.dir_path, "rdkit")
        create_dir(rdkit_dir, self._logger)

        self._logger.debug(
            "Generating conformers and computing energies using RDKit",
        )
        try:
            mol: Mol = AddHs(
                get_mol(
                    StandardizeSmiles(self.smiles),
                    self._logger,
                ),
            )
        except Exception as err:
            self._logger.exception(err)
            raise

        n_conformers: int = self._config["rdkit"]["n_conformers"]

        self._logger.debug(
            f"Generating {n_conformers} conformers",
        )
        conformer_ids: list[int] = EmbedMultipleConfs(
            mol,
            n_conformers,
            useRandomCoords=True,
            randomSeed=self._config["rdkit"]["random_seed"],
            boxSizeMult=self._config["rdkit"]["conformer_box_size"],
            numThreads=self._config["rdkit"]["n_cores"],
        )

        self._logger.debug(f"Optimising the {n_conformers} conformers")

        MMFFOptimizeMoleculeConfs(
            mol,
            numThreads=self._config["rdkit"]["n_cores"],
            maxIters=self._config["rdkit"]["mmff_iters"],
            mmffVariant=self._config["rdkit"]["mmff"],
        )

        self._logger.debug("Computing molecular properties for MMFF")
        mp: MMFFMolProperties = MMFFGetMoleculeProperties(
            mol,
            mmffVariant=self._config["rdkit"]["mmff"],
            mmffVerbosity=int(self._args.verbose),
        )

        self._logger.debug(f"Computing energies for the {n_conformers} conformers")
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

        self._logger.debug(
            f"Aligning the {n_conformers} conformers by their geometries",
        )
        AlignMolConformers(mol, maxIters=self._config["rdkit"]["align_iters"])

        self._logger.debug("Finished generating conformers")

        sdf_file: Path = Path(rdkit_dir, "conformers.sdf")
        self._logger.debug(f"Writing conformers and their energies to {sdf_file}")

        try:
            with open(sdf_file, "w") as fstream:
                writer: SDWriter = SDWriter(fstream)
                for cid, energy in conformer_energies:
                    mol.SetIntProp("conformer_id", cid)
                    mol.SetDoubleProp("energy", energy)
                    writer.write(mol, confId=cid)
                writer.close()
        except Exception as err:
            self._logger.exception(err)
            self._logger.exception(f"Failed to write RDKit conformers to {sdf_file}")
            raise
        else:
            self._logger.debug("Finished writing conformers and their energies")

        # Write conformers to .xyz files
        self._logger.debug(f"Getting conformer coordinates from {sdf_file}")
        sdfstream: IOBase = open(sdf_file, "rb")
        mol_supplier: ForwardSDMolSupplier = ForwardSDMolSupplier(
            fileobj=sdfstream,
            sanitize=False,
            removeHs=False,
        )

        # Running xTB for all conformers
        self._logger.debug("Computing thermodynamic parameters of conformers using xTB")

        xtb_default_args: list[str] = parse_command_args(
            sub_config=["xtb", "rdkit"],
            config=self._config,
            logger=self._logger,
        )

        for conformer in mol_supplier:
            # Conformer id and location of xyz file
            conformer_id: int = conformer.GetIntProp("conformer_id")
            conformer_xyz_file: Path = Path(
                rdkit_dir,
                f"conformer_{conformer_id}.xyz",
            )

            try:
                MolToXYZFile(
                    mol=mol,
                    filename=conformer_xyz_file,
                    precision=self._config["rdkit"]["precision"],
                )
            except Exception as err:
                self._logger.exception(err)
                self._logger.exception(
                    f"Failed to write conformer coordinates to {conformer_xyz_file}",
                )
                raise

            # Creating output dir for xTB thermo calculation
            xtb_dir: Path = Path(
                self.dir_path,
                "xtb",
                "rdkit",
                f"conformer_{conformer_id}",
            )
            create_dir(xtb_dir, self._logger)

            # Specify location for xtb log file
            xtb_output_file: Path = Path(
                xtb_dir,
                f"conformer_{conformer_id}.txt",
            )

            # Optimisation with xTB
            xtb_conformer_args: list[str] = [
                "xtb",
                str(conformer_xyz_file.absolute()),
                "--namespace",
                f"conformer_{conformer_id}",
            ]
            xtb_args: list[str] = xtb_conformer_args + xtb_default_args

            try:
                with open(xtb_output_file, "w") as fstream:
                    subprocess.Popen(
                        args=xtb_args,
                        stdout=fstream,
                        stderr=subprocess.STDOUT,
                        cwd=xtb_dir,
                    ).wait()
            except Exception as err:
                self._logger.exception(err)
                self._logger.exception(
                    f"Failed to run xTB command with arguments {xtb_args}",
                )
        sdfstream.close()
        self._logger.debug("Finished running xTB on conformers")

        self._logger.debug("Compiling results of xTB computations")

        xtb_results: dict[str, list[float | None]] = {"conformer_id": []}
        xtb_results |= {param: [] for param in THERMODYNAMIC_PARAMS}

        xtb_dir: Path = Path(self.dir_path, "xtb", "rdkit")
        conformer_dirs: list[Path] = [d for d in xtb_dir.iterdir() if d.is_dir()]
        self._logger.debug(f"Searching conformer dirs {conformer_dirs}")

        for conformer_dir in conformer_dirs:
            xtb_output_file: Path = Path(
                conformer_dir,
                f"{conformer_dir.name}.txt",
            )

            try:
                xtb_output = parse_xtb_output(
                    xtb_output_file=xtb_output_file,
                    temperature=self._config["xtb"]["rdkit"]["etemp"],
                    logger=self._logger,
                )
            except Exception:
                self._logger.exception(
                    f"Failed to parse xtb results for conformer in {conformer_dir.name}",
                )
                continue
            else:
                xtb_results["conformer_id"].append(
                    int(conformer_dir.name.split("_")[1]),
                )
                for param in THERMODYNAMIC_PARAMS:
                    xtb_results[param].append(xtb_output[param])

        xtb_results_file: Path = Path(xtb_dir, "results.csv")
        self._logger.debug(f"Writing results to {xtb_results_file}")

        xtb_results_df = pd.DataFrame(xtb_results, dtype=np.float64).sort_values(
            "G",
        )
        xtb_results_df.to_csv(xtb_results_file, header=True, index=False)

        self._logger.debug(
            f"Finished writing results for {self.__class__.__name__} ({self.smiles})",
        )

        # add to self.results
        best_rdkit_conformer_results = xtb_results_df.iloc[0].to_dict()
        del best_rdkit_conformer_results["conformer_id"]
        self._results |= {"rdkit": best_rdkit_conformer_results}

    def _run_crest_confgen(self) -> None:
        """
        Running the CREST standard procedure to generate a conformer/rotamer ensemble.
        Subsequently performing thermo calculations using xTB on best conformer (if desired).
        """
        crest_dir: Path = Path(self.dir_path, "crest_confgen")
        create_dir(crest_dir, self._logger)

        xtb_rdkit_dir: Path = Path(self.dir_path, "xtb", "rdkit")
        xtb_rdkit_results_file: Path = Path(
            xtb_rdkit_dir,
            "results.csv",
        )
        best_rdkit_conformer_results: dict[str, float | None] = (
            parse_best_rdkit_conformer(xtb_rdkit_results_file)
        )
        if best_rdkit_conformer_results["conformer_id"] is None:
            raise NoResultsError(
                "Failed to extract best RDKit conformer for crest_confgen",
            )
        best_rdkit_conformer_id: int = int(best_rdkit_conformer_results["conformer_id"])

        best_rdkit_conformer_xyz_file: Path = Path(
            xtb_rdkit_dir,
            f"conformer_{best_rdkit_conformer_id}",
            f"conformer_{best_rdkit_conformer_id}.xtbopt.xyz",
        )

        # specify location for crest log file
        crest_output_file: Path = Path(crest_dir, "crest.txt")

        # conformer generation with CREST
        crest_args: list[str] = [
            "crest",
            str(best_rdkit_conformer_xyz_file.absolute()),
        ]
        crest_args += parse_command_args(
            sub_config=["crest", "confgen"],
            config=self._config,
            logger=self._logger,
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
            self._logger.exception(err)
            self._logger.exception(f"Failed to run CREST with arguments {crest_args}")
            raise

        if self._config["steps"]["xtb"]:
            best_crest_conformer_xyz_file: Path = Path(
                crest_dir,
                "crest_best.xyz",
            )
            xtb_dir: Path = Path(self.dir_path, "xtb", "crest")
            create_dir(xtb_dir, self._logger)
            xtb_results: dict[str, float | None] = self._run_xtb(
                xtb_input_file=best_crest_conformer_xyz_file,
                xtb_dir=xtb_dir,
                xtb_sub_config=["xtb", "crest"],
            )

            # add to self.results
            self._results |= {"crest": xtb_results}

    def _run_censo(self) -> None:
        """
        Running CENSO to DFT refine the conformer ensemble from CREST conformer generation.
        """

        # create .censo2rc for run
        create_censorc(args=self._args, logger=self._logger)

        censo_dir: Path = Path(self.dir_path, "censo")
        create_dir(censo_dir, self._logger)

        # specify location for censo log file
        censo_output_file: Path = Path(censo_dir, "censo.txt")
        censo_config_file: Path = Path(
            self._args.config,
            "temp",
            ".censo2rc",
        )

        # specify location for crest conformers file
        crest_conformers_file: Path = Path(
            self.dir_path,
            "crest_confgen",
            "crest_conformers.xyz",
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
            config=self._config,
            logger=self._logger,
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
            self._logger.exception(err)
            self._logger.exception(f"Failed to run CENSO with arguments {censo_args}")
            raise

        censo_results: dict[str, dict[str, float | None]] = parse_censo_output(
            censo_output_file=censo_output_file,
            temperature=self._config["censo"]["general"]["temperature"],
            logger=self._logger,
        )

        censo_results_file: Path = Path(
            censo_dir,
            "results.json",
        )

        self._logger.debug(f"Writing results to {censo_results_file}")

        with open(censo_results_file, "w") as fstream:
            json.dump(
                censo_results,
                fstream,
                sort_keys=True,
                allow_nan=True,
                indent=4,
            )

        self._results |= censo_results

    def _run_xtb(
        self,
        xtb_input_file: Path,
        xtb_dir: Path,
        xtb_sub_config: list[str],
    ) -> dict[str, float | None]:
        # Running xTB within intermediate pipeline sections
        xtb_args: list[str] = [
            "xtb",
            str(xtb_input_file.absolute()),
            "--namespace",
            xtb_input_file.name.split(".")[0],
        ]
        xtb_args += parse_command_args(
            sub_config=xtb_sub_config,
            config=self._config,
            logger=self._logger,
        )

        xtb_output_file: Path = Path(xtb_dir, "xtb.txt")
        try:
            with open(xtb_output_file, "w") as fstream:
                subprocess.Popen(
                    args=xtb_args,
                    stdout=fstream,
                    stderr=subprocess.STDOUT,
                    cwd=xtb_dir,
                ).wait()
        except Exception as err:
            self._logger.exception(err)
            self._logger.exception(
                f"Failed to run xTB with arguments {xtb_args}",
            )
            raise

        try:
            xtb_results: dict[str, float | None] = parse_xtb_output(
                xtb_output_file=xtb_output_file,
                temperature=self._config["xtb"][xtb_sub_config[1]]["etemp"],
                logger=self._logger,
            )
        except Exception:
            self._logger.exception(
                f"Failed to parse xtb results in file {xtb_output_file}",
            )
            raise

        xtb_results_file: Path = Path(xtb_dir, "results.json")
        self._logger.debug(f"Writing results to {xtb_results_file}")

        with open(xtb_results_file, "w") as fstream:
            json.dump(
                xtb_results,
                fstream,
                sort_keys=True,
                allow_nan=True,
                indent=4,
            )

        return xtb_results

    def _run_crest_entropy(self) -> None:
        """
        Running CREST to estimate the conformational entropy of the ensemble.
        """
        # extract best CENSO conformer from 3_REFINEMENT
        censo_dir: Path = Path(self.dir_path, "censo")
        censo_output_file: Path = Path(censo_dir, "censo.txt")
        censo_conformers_xyz_file: Path = Path(
            censo_dir,
            "3_REFINEMENT.xyz",
        )
        best_censo_conformer_xyz_file: Path = Path(
            censo_dir,
            "censo_best.xyz",
        )
        try:
            best_censo_conformer: str = parse_best_censo_conformers(
                censo_output_file=censo_output_file,
                logger=self._logger,
            )["3_REFINEMENT"]
            extract_best_conformer_xyz(
                censo_conformers_xyz_file=censo_conformers_xyz_file,
                best_conformer_id=best_censo_conformer,
                output_xyz_file=best_censo_conformer_xyz_file,
                logger=self._logger,
            )
        except Exception as err:
            self._logger.exception(err)
            raise NoResultsError(
                "Failed to extract best CENSO conformer for crest_entropy",
            ) from err

        # setup crest_entropy directory
        crest_dir: Path = Path(self.dir_path, "crest_entropy")
        create_dir(crest_dir, self._logger)

        # specify location for crest log file
        crest_output_file: Path = crest_dir / "crest.txt"

        # conformer generation with CREST
        crest_args: list[str] = [
            "crest",
            str(best_censo_conformer_xyz_file.absolute()),
        ]
        crest_args += parse_command_args(
            sub_config=["crest", "entropy"],
            config=self._config,
            logger=self._logger,
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
            self._logger.exception(err)
            self._logger.exception(f"Failed to run CREST with arguments {crest_args}")
            raise

        crest_results: dict[str, float | None] = parse_crest_entropy_output(
            crest_output_file=crest_output_file,
            logger=self._logger,
        )

        if self._config["steps"]["censo"]:
            censo_corrected_results: dict[str, float | None] = self._results[
                "3_REFINEMENT"
            ].copy()
            if (censo_corrected_results["S"] is None) or (
                crest_results["S_total"] is None
            ):
                raise NoResultsError(
                    "Failed to apply CREST entropy correction to CENSO results",
                )
            censo_corrected_results["S"] += crest_results["S_total"]
            self._results["censo_corrected"] = censo_corrected_results

        crest_results_file: Path = crest_dir / "results.json"
        self._logger.debug(f"Writing results to {crest_results_file}")

        with open(crest_results_file, "w") as fstream:
            json.dump(
                crest_results,
                fstream,
                sort_keys=True,
                allow_nan=True,
                indent=4,
            )

    def load_final_results(self) -> bool:
        """Checks whether final results already exist.

        If so, the LaREST pipeline does not have to be run for the molecule.

        Returns
        -------
        bool
            Whether final results already exist

        """
        results_file: Path = Path(self.dir_path, "results.json")
        if results_file.exists():
            try:
                with open(results_file) as fstream:
                    results: dict[str, dict[str, float | None]] = json.load(fstream)
                    self._results |= results
            except Exception as err:
                self._logger.exception(err)
                self._logger.exception(
                    f"Failed to load pre-existing final results from {results_file}, will run LaREST pipeline",
                )
                return False
            else:
                self._logger.info(
                    f"Loaded pre-existing final results from {results_file}",
                )
                self._logger.debug(f"Pre-existing final results:\n {results}")
                return True
        else:
            self._logger.info(
                f"No pre-existing final results detected in {results_file}, will run LaREST pipeline",
            )
            self._logger.info("Pipeline will be run for molecule")

            return False

    def _write_final_results(self) -> None:
        results_file: Path = Path(self.dir_path, "results.json")

        self._logger.debug(f"Writing final results to {results_file}")

        with open(results_file, "w") as fstream:
            json.dump(
                self._results,
                fstream,
                sort_keys=True,
                indent=4,
                allow_nan=True,
            )

    def _cleanup_pipeline(self) -> None:
        # Remove temp dirs
        temp_config_dir: Path = Path(self._args.config, "temp")
        remove_dir(temp_config_dir, self._logger)


class Initiator(LarestMol):
    def __init__(
        self,
        smiles: str,
        args: Namespace,
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
            Path(self._args.output)
            / self.__class__.__name__
            / f"{slugify(self._smiles)}_{self._length}"
        )

    @LarestMol.smiles.getter
    def smiles(self) -> str:
        try:
            return build_polymer(
                monomer_smiles=self._smiles,
                polymer_length=self._length,
                reaction_type=self._config["reaction"]["type"],
                config=self._config,
                logger=self._logger,
            )
        except Exception:
            self._logger.exception("Failed to build polymer smiles")
            raise

    def __init__(
        self,
        smiles: str,
        length: int,
        args: Namespace,
        config: dict[str, Any],
    ) -> None:
        self._length = length
        super().__init__(smiles=smiles, args=args, config=config)


class Monomer(LarestMol):
    _ring_size: int | None
    _initiator: Initiator
    _polymers: list[Polymer]

    @property
    def ring_size(self) -> int | None:
        return get_ring_size(
            smiles=self.smiles,
            logger=self._logger,
        )

    @property
    def initiator(self) -> Initiator:
        return self._initiator

    @property
    def polymers(self) -> list[Polymer]:
        return self._polymers

    def __init__(
        self,
        smiles: str,
        args: Namespace,
        config: dict[str, Any],
    ) -> None:
        super().__init__(smiles=smiles, args=args, config=config)

        self._initiator = Initiator(
            smiles=config["reaction"]["initiator"],
            args=self._args,
            config=self._config,
        )
        self._polymers = [
            Polymer(
                smiles=self.smiles,
                length=length,
                args=self._args,
                config=self._config,
            )
            for length in self._config["reaction"]["lengths"]
        ]

    def compile_results(self) -> None:
        summary_dir: Path = self.dir_path / "summary"
        create_dir(summary_dir, self._logger)

        # iterating over pipeline sections
        for section in self._results:
            # creating headings for final summary csv
            summary_headings: list[str] = ["polymer_length"]
            for mol_type in ["monomer", "initiator", "polymer"]:
                summary_headings.extend(
                    [f"{mol_type}_{param}" for param in THERMODYNAMIC_PARAMS],
                )

            # final summary table
            summary: dict[str, list[float | None]] = {
                heading: [] for heading in summary_headings
            }

            # iterating over polymer lengths
            for polymer in self._polymers:
                # adding monomer, initiator, and polymer results to summary
                summary["polymer_length"].append(polymer.length)
                for param in THERMODYNAMIC_PARAMS:
                    summary[f"polymer_{param}"].append(
                        polymer.results[section][param],
                    )
                    summary[f"monomer_{param}"].append(
                        self.results[section][param],
                    )
                    summary[f"initiator_{param}"].append(
                        self._initiator.results[section][param]
                        if (self._config["reaction"]["type"] == "ROR")
                        else 0,
                    )

            # converting to df
            summary_df: pd.DataFrame = pd.DataFrame(
                summary,
                index=None,
                dtype=np.float64,
            ).sort_values(
                "polymer_length",
                ascending=True,
            )

            # calculating deltas
            # TODO: try other calculation methods
            for param in THERMODYNAMIC_PARAMS:
                summary_df[f"delta_{param}"] = (
                    summary_df[f"polymer_{param}"]
                    - (summary_df["polymer_length"] * summary_df[f"monomer_{param}"])
                    - summary_df[f"initiator_{param}"]
                ) / summary_df["polymer_length"]

            # specify to save results
            summary_file: Path = summary_dir / f"{section}.csv"

            summary_df.to_csv(
                summary_file,
                header=True,
                index=False,
            )
