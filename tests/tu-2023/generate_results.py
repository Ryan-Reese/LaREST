from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from larest.base import Monomer
from larest.constants import ENTHALPY_PLOTTING_SECTIONS
from larest.parsers import LarestArgumentParser
from larest.setup import get_config


def generate_results(
    args: Namespace,
    config: dict[str, Any],
):
    # get experimental results
    experimental_results_file: Path = Path("./experimental.csv")
    experimental_results: pd.DataFrame = pd.read_csv(
        experimental_results_file,
        index_col="name",
        comment="#",
    )

    monomer_names: list[str] = []
    predicted_h: list[float | None] = []
    predicted_s: list[float | None] = []

    # iterate over monomers
    for monomer_idx, monomer_smiles in enumerate(config["reaction"]["monomers"]):
        monomer: Monomer = Monomer(
            smiles=monomer_smiles,
            args=args,
            config=config,
        )
        monomer_names.append(experimental_results.index[monomer_idx])

        summary_dir: Path = Path(monomer.dir_path, "summary")

        # check if results exist
        if not summary_dir.exists():
            predicted_h.append(None)
            predicted_s.append(None)
            continue

        # obtain final pipeline result for enthalpy
        section = ENTHALPY_PLOTTING_SECTIONS[-1]
        results_file: Path = summary_dir / f"{section}.csv"

        results: pd.DataFrame = pd.read_csv(
            results_file,
            header=0,
            dtype=np.float64,
        )

        x = (1 / results["polymer_length"]).to_numpy()[::-1]
        y = (results["delta_H"] / 1000).to_numpy()[::-1]

        h_regressor = LinearRegression(fit_intercept=True)
        h_regressor.fit(x.reshape(-1, 1), y)

        predicted_h.append(h_regressor.intercept_)

        # obtain final pipeline result for entropy
        section = ENTHALPY_PLOTTING_SECTIONS[-1]
        results_file: Path = summary_dir / f"{section}.csv"

        results: pd.DataFrame = pd.read_csv(
            results_file,
            header=0,
            dtype=np.float64,
        )

        x = (1 / results["polymer_length"]).to_numpy()[::-1]
        y = (results["delta_S"]).to_numpy()[::-1]

        s_regressor = LinearRegression(fit_intercept=True)
        s_regressor.fit(x.reshape(-1, 1), y)

        predicted_s.append(s_regressor.intercept_)

        # write final predicted results
        predicted_results: pd.DataFrame = pd.DataFrame(
            {
                "name": monomer_names,
                "delta_H": predicted_h,
                "delta_S": predicted_s,
            },
        )

        predicted_results.to_csv(
            "./computational.csv",
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

    generate_results(
        args=args,
        config=config,
    )
