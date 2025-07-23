import os
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from larest.helpers.chem import get_ring_size
from larest.helpers.parsers import XTBParser

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


def compute_delta_limit(monomer_smiles, args):
    step1_dir = os.path.join(args.output, "step1")
    monomer_results_file = os.path.join(
        step1_dir,
        f"results_{monomer_smiles}.csv",
    )
    monomer_results = pd.read_csv(monomer_results_file, header=0, dtype=np.float64)

    h_regressor = LinearRegression(fit_intercept=True)
    s_regressor = LinearRegression(fit_intercept=True)

    h_regressor.fit(
        X=(1 / monomer_results["polymer_length"])
        .to_numpy(dtype=np.float64)
        .reshape(-1, 1),
        y=monomer_results["delta_h"].to_numpy(dtype=np.float64),
    )
    s_regressor.fit(
        X=(1 / monomer_results["polymer_length"])
        .to_numpy(dtype=np.float64)
        .reshape(-1, 1),
        y=monomer_results["delta_s"].to_numpy(dtype=np.float64),
    )

    return h_regressor.intercept_, s_regressor.intercept_


def plot_extrapolation(
    args,
    config,
    save_path="./tests/plot_vertical/extrapolation.svg",
):
    step1_dir = os.path.join(args.output, "step1")

    fig, ax = plt.subplots()

    for monomer_smiles in config["reaction"]["monomers"]:
        monomer_ring_size = get_ring_size(monomer_smiles)
        monomer_results_file = os.path.join(step1_dir, f"results_{monomer_smiles}.csv")
        monomer_results = pd.read_csv(monomer_results_file, header=0, dtype=np.float64)

        X = (1 / monomer_results["polymer_length"]).to_numpy()[::-1]
        y = (monomer_results["delta_h"] / 1000).to_numpy()[::-1]

        h_regressor = LinearRegression(fit_intercept=True)

        h_regressor.fit(X.reshape(-1, 1), y)
        print(h_regressor.score(X.reshape(-1, 1), y))

        ax.scatter(X, y, label=f"ringsize={monomer_ring_size}")
        best_fit_x = np.linspace(X[0], X[-1], num=50)
        best_fit_y = h_regressor.predict(best_fit_x.reshape(-1, 1))
        ax.plot(
            best_fit_x,
            best_fit_y,
            linestyle="-",
            label=f"ringsize={monomer_ring_size}",
            alpha=0.5,
        )

    ax.set_ylabel("Enthalpy of Polymerisation (kJmol^-1)")
    ax.set_xlabel("1/L")
    ax.set_xlim(right=1.25)

    handles, labels = ax.get_legend_handles_labels()
    unique_handles_labels = [
        (handle, label)
        for i, (handle, label) in enumerate(zip(handles, labels))
        if label not in labels[:i]
    ]
    fig.legend(*zip(*unique_handles_labels), loc="outside center right", ncols=1)

    plt.savefig(save_path)


def plot_ring_size(args, config, save_path="./tests/plot_vertical/extrapolation.svg"):
    results = dict(ring_size=[], delta_h=[], delta_s=[])

    fig, ax = plt.subplots(layout="constrained")

    for monomer_smiles in config["reaction"]["monomers"]:
        monomer_ring_size = get_ring_size(monomer_smiles)
        monomer_delta_h, monomer_delta_s = compute_delta_limit(
            monomer_smiles=monomer_smiles,
            args=args,
        )
        results["ring_size"].append(monomer_ring_size)
        results["delta_h"].append(monomer_delta_h / 1000)  # convert to kJ/mol
        results["delta_s"].append(monomer_delta_s)

    df = pd.DataFrame(data=results, dtype=np.float64).sort_values(
        "ring_size", ascending=True
    )

    for label in ["delta_h", "delta_s"]:
        line = ax.plot(
            df["ring_size"],
            df[label],
            label=label,
            linestyle="-",
            marker="o",
        )

    ax.set_ylabel("Enthalpy (kJmol^-1)/Entropy (Jmol^-1K^-1)of Polymerisation")
    ax.set_xlabel("Monomer Ring Size")
    ax.legend(loc="upper right", ncols=1)

    plt.savefig(save_path)


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser = XTBParser()
    args = parser.parse_args()

    # load step 1 config
    config_file = os.path.join(args.config, "config.toml")

    with open(config_file, "rb") as fstream:
        config = tomllib.load(fstream)

    plot_ring_size(
        args=args,
        config=config,
        save_path="./tests/plot_vertical/ring_size.svg",
    )
    plot_extrapolation(
        args=args,
        config=config,
        save_path="./tests/plot_vertical/extrapolation.svg",
    )
