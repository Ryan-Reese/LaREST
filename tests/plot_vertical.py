import os
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from larest.config.parsers import XTBParser
from larest.helpers import get_ring_size

"""
    This script intends to generate a plot of polymerisation enthalpy
    against monomer chain size, using a standard LaREST run as input.
    Currently, this script uses thermodynamic parameters outputted from
    Step 1 of the pipeline, which only performs rough optimisation using xTB.
    The monomers in this case are all unsubstituted lactones.

    Secondarily, this script generates a large plot, where each subplot
    shows the linear regression used to approximate the polymerisation enthalpy
    of the respective monomer.
"""

# TODO: simultaneously plot against experimentally-validated results


def compute_polymerisation_enthalpy(monomer_smiles, args, config):
    step1_dir = os.path.join(args.output, "step1")
    monomer_results_file = os.path.join(step1_dir, f"results_{monomer_smiles}.csv")
    monomer_results = pd.read_csv(monomer_results_file, header=0, dtype=np.float64)

    regressor = LinearRegression(fit_intercept=True)

    regressor.fit(
        X=(1 / monomer_results["polymer_length"])
        .to_numpy(dtype=np.float64)
        .reshape(-1, 1),
        y=monomer_results["delta_h"].to_numpy(dtype=np.float64),
    )

    return regressor.intercept_


def fit_func(x, a, b, c, d, e):
    return a * (x**4) + b * (x**3) + c * (x**2) + d * (x**1) + e


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser = XTBParser()
    args = parser.parse_args()

    # load step 1 config
    step1_config_file = os.path.join(args.config, "step1.toml")

    with open(step1_config_file, "rb") as fstream:
        config = tomllib.load(fstream)

    input_file = os.path.join(args.config, "input.txt")

    with open(input_file, "r") as fstream:
        monomer_smiles_list = fstream.read().splitlines()

    results = dict(ring_size=[], enthalpy=[])

    for monomer_smiles in monomer_smiles_list:
        monomer_ring_size = get_ring_size(monomer_smiles)
        monomer_enthalpy = compute_polymerisation_enthalpy(
            monomer_smiles=monomer_smiles, args=args, config=config
        )
        results["ring_size"].append(monomer_ring_size)
        results["enthalpy"].append(monomer_enthalpy / 1000)

    df = pd.DataFrame(data=results, dtype=np.float64).sort_values("ring_size")

    fig, ax = plt.subplots(layout="constrained")

    # Top figure
    line = ax.plot(
        df["ring_size"],
        df["enthalpy"],
        label="Unsubstituted lactones",
        color="tab:blue",
        linestyle="-",
        marker="o",
    )

    # add curved fit
    best_fit = curve_fit(
        f=fit_func, xdata=results["ring_size"], ydata=results["enthalpy"]
    )
    x = np.linspace(df["ring_size"].iloc[0], df["ring_size"].iloc[-1], 50)
    best_fit_line = ax.plot(
        x,
        fit_func(x, *best_fit[0]),
        label="Best fit line",
        color="tab:blue",
        linestyle="-",
        alpha=0.5,
    )

    ax.set_ylabel("Enthalpy of Polymerisation (kJmol^-1)")
    ax.set_xlabel("Monomer Ring Size")
    # ax.set_ylim(0, 400)
    ax.legend(loc="upper right", ncols=1)

    plt.show()
    # plt.savefig("./tests/plot_vertical.svg")
