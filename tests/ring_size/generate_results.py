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
    # iterate over LaREST sections
    for section in ENTHALPY_PLOTTING_SECTIONS:
        ring_sizes: list[int | None] = []
        predicted_h: list[float | None] = []

        # iterate over monomers
        for monomer_smiles in config["reaction"]["monomers"]:
            monomer: Monomer = Monomer(
                smiles=monomer_smiles,
                args=args,
                config=config,
            )
            ring_sizes.append(monomer.ring_size)

            results_file: Path = Path(
                monomer.dir_path,
                "summary",
                f"{section}.csv",
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

            x = (1 / results["polymer_length"]).to_numpy()[::-1]
            y = (results["delta_H"] / 1000).to_numpy()[::-1]

            h_regressor = LinearRegression(fit_intercept=True)
            h_regressor.fit(x.reshape(-1, 1), y)

            predicted_h.append(h_regressor.intercept_)

            # write final predicted results
            predicted_results: pd.DataFrame = pd.DataFrame(
                {
                    "ring_size": ring_sizes,
                    "delta_H": predicted_h,
                },
            )

            predicted_results.to_csv(
                f"./data/computational_{section}.csv",
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
