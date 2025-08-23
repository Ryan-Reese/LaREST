from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from sklearn.linear_model import LinearRegression

from larest.base import Monomer
from larest.constants import ENTHALPY_PLOTTING_SECTIONS, ENTROPY_PLOTTING_SECTIONS
from larest.parsers import LarestArgumentParser
from larest.setup import get_config


def generate_larest_results(
    args: Namespace,
    config: dict[str, Any],
):
    # iterate over LaREST sections
    for section_name in ENTHALPY_PLOTTING_SECTIONS:
        monomer_smiles: list[str | None] = []
        predicted_h: list[float | None] = []

        # iterate over monomers
        for smiles in config["reaction"]["monomers"]:
            monomer: Monomer = Monomer(
                smiles=smiles,
                args=args,
                config=config,
            )
            monomer_smiles.append(StandardizeSmiles(monomer.smiles))

            results_file: Path = Path(
                monomer.dir_path,
                "summary",
                f"{section_name}.csv",
            )

            # check if results exist
            if not results_file.exists():
                predicted_h.append(None)
                continue

            results: pd.DataFrame = pd.read_csv(
                results_file,
                header=0,
                dtype=np.float64,
            )

            if results["delta_H"].hasnans:
                predicted_h.append(None)
                continue

            x = (1 / results["polymer_length"]).to_numpy()[::-1]
            y = (results["delta_H"] / 1000).to_numpy()[::-1]

            h_regressor = LinearRegression(fit_intercept=True)
            h_regressor.fit(x.reshape(-1, 1), y)

            predicted_h.append(h_regressor.intercept_)

        # write final predicted results
        predicted_results: pd.DataFrame = pd.DataFrame(
            {
                "monomer_smiles": monomer_smiles,
                "delta_H": predicted_h,
            },
        )

        predicted_results.to_csv(
            f"./data/{section_name}_H.csv",
            na_rep="null",
            header=True,
            index=False,
        )
    for section_name in ENTROPY_PLOTTING_SECTIONS:
        monomer_smiles: list[int | None] = []
        predicted_s: list[float | None] = []

        # iterate over monomers
        for smiles in config["reaction"]["monomers"]:
            monomer: Monomer = Monomer(
                smiles=smiles,
                args=args,
                config=config,
            )
            monomer_smiles.append(StandardizeSmiles(monomer.smiles))

            results_file: Path = Path(
                monomer.dir_path,
                "summary",
                f"{section_name}.csv",
            )

            # check if results exist
            if not results_file.exists():
                predicted_s.append(None)
                continue

            results: pd.DataFrame = pd.read_csv(
                results_file,
                header=0,
                dtype=np.float64,
            )

            if results["delta_S"].hasnans:
                predicted_s.append(None)
                continue

            x = (1 / results["polymer_length"]).to_numpy()[::-1]
            y = (results["delta_S"]).to_numpy()[::-1]

            s_regressor = LinearRegression(fit_intercept=True)
            s_regressor.fit(x.reshape(-1, 1), y)

            predicted_s.append(s_regressor.intercept_)

        # write final predicted results
        predicted_results: pd.DataFrame = pd.DataFrame(
            {
                "monomer_smiles": monomer_smiles,
                "delta_S": predicted_s,
            },
        )

        predicted_results.to_csv(
            f"./data/{section_name}_S.csv",
            na_rep="null",
            header=True,
            index=False,
        )


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser: LarestArgumentParser = LarestArgumentParser()
    args: Namespace = parser.parse_args()

    # load LaREST config
    config: dict[str, Any] = get_config(args=args)

    generate_larest_results(
        args=args,
        config=config,
    )
