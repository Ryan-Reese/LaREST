from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from sklearn.linear_model import LinearRegression

from larest.base import Monomer
from larest.constants import ENTHALPY_PLOTTING_SECTIONS, ENTROPY_PLOTTING_SECTIONS
from larest.parsers import LarestArgumentParser
from larest.setup import get_config


def plot_accuracy(
    args: Namespace,
    config: dict[str, Any],
    output_dir: Path,
):
    # get experimental results
    experimental_results: pd.DataFrame = pd.read_csv(
        Path(
            "./data",
            "tropic.csv",
        ),
        index_col=None,
        comment="#",
    )

    # first index is pipeline section, second index is ring size
    h_absolute_errors: list[list[float]] = [
        [] for _ in range(len(ENTHALPY_PLOTTING_SECTIONS))
    ]
    s_absolute_errors: list[list[float]] = [
        [] for _ in range(len(ENTROPY_PLOTTING_SECTIONS))
    ]

    # iterate over monomers
    for monomer_smiles in config["reaction"]["monomers"]:
        monomer: Monomer = Monomer(
            smiles=monomer_smiles,
            args=args,
            config=config,
        )

        summary_dir: Path = Path(monomer.dir_path, "summary")

        # check if results exist
        if not summary_dir.exists():
            continue

        # iterating over sections
        for section_idx, section in enumerate(ENTHALPY_PLOTTING_SECTIONS):
            results_file: Path = summary_dir / f"{section}.csv"

            results: pd.DataFrame = pd.read_csv(
                results_file,
                header=0,
                dtype=np.float64,
            )

            # check if results fully exist
            if results["delta_H"].hasnans:
                continue

            x = (1 / results["polymer_length"]).to_numpy()[::-1]
            y = (results["delta_H"] / 1000).to_numpy()[::-1]

            h_regressor = LinearRegression(fit_intercept=True)
            h_regressor.fit(x.reshape(-1, 1), y)

            predicted_h = h_regressor.intercept_
            try:
                experimental_h = experimental_results.loc[
                    experimental_results["monomer_smiles"]
                    == StandardizeSmiles(monomer_smiles)
                ]["delta_H"].item()
                h_absolute_errors[section_idx].append(abs(predicted_h - experimental_h))
            except ValueError:
                h_absolute_errors[section_idx].append(abs(predicted_h))

        # iterating over sections
        for section_idx, section in enumerate(ENTROPY_PLOTTING_SECTIONS):
            results_file: Path = summary_dir / f"{section}.csv"

            results: pd.DataFrame = pd.read_csv(
                results_file,
                header=0,
                dtype=np.float64,
            )

            # check if results fully exist
            if results["delta_S"].hasnans:
                continue

            x = (1 / results["polymer_length"]).to_numpy()[::-1]
            y = (results["delta_S"]).to_numpy()[::-1]

            s_regressor = LinearRegression(fit_intercept=True)
            s_regressor.fit(x.reshape(-1, 1), y)

            predicted_s = s_regressor.intercept_
            try:
                experimental_s = experimental_results.loc[
                    experimental_results["monomer_smiles"]
                    == StandardizeSmiles(monomer_smiles)
                ]["delta_S"].item()
                s_absolute_errors[section_idx].append(abs(predicted_s - experimental_s))
            except ValueError:
                s_absolute_errors[section_idx].append(abs(predicted_s))

    _, ax = plt.subplots(1, 1)

    h_section_labels = [
        "RDKit",
        "CREST",
        "CENSO",
    ]
    s_section_labels = [
        "RDKit",
        "CREST",
        "CENSO",
        r"CENSO w/ $S_{conf}$",
    ]

    h_mean_absolute_errors: list[float] = [
        sum(errors) / len(errors) for errors in h_absolute_errors
    ]
    h_mean_absolute_errors.append(h_mean_absolute_errors[-1])

    h_std_absolute_errors: list[float] = [
        np.std(errors).item() for errors in h_absolute_errors
    ]
    h_std_absolute_errors.append(h_std_absolute_errors[-1])

    s_mean_absolute_errors: list[float] = [
        sum(errors) / len(errors) for errors in s_absolute_errors
    ]
    s_std_absolute_errors: list[float] = [
        np.std(errors).item() for errors in s_absolute_errors
    ]

    # plot errors
    x = np.arange(len(ENTROPY_PLOTTING_SECTIONS))
    ax.errorbar(
        x,
        h_mean_absolute_errors,
        yerr=h_std_absolute_errors,
        ecolor="tab:gray",
        elinewidth=(0.8 * mpl.rcParams["lines.linewidth"]),
        capsize=4.0,
        color="tab:gray",
        linestyle="dashed",
        linewidth=(1.2 * mpl.rcParams["lines.linewidth"]),
        marker="s",
        markersize=(0.5 * (mpl.rcParams["lines.markersize"] ** 2)),
    )

    # ax2 = ax.twinx()
    # ax2.errorbar(
    #     x,
    #     s_mean_absolute_errors,
    #     yerr=s_std_absolute_errors,
    #     ecolor="tab:gray",
    #     elinewidth=(0.8 * mpl.rcParams["lines.linewidth"]),
    #     capsize=4.0,
    #     color="tab:gray",
    #     linestyle="dashdot",
    #     linewidth=(1.2 * mpl.rcParams["lines.linewidth"]),
    #     marker="s",
    #     markersize=(mpl.rcParams["lines.markersize"] ** 2),
    # )
    #
    ax.set_ylabel(r"$\Delta H_{p}$ MAE $(\mathrm{kJ mol^{-1}})$")
    # ax2.set_ylabel(r"$\Delta S_{p}$ MAE $(\mathrm{J mol^{-1}})$")
    ax.set_ylim(bottom=0)
    # ax2.set_ylim(bottom=0)

    # ax2 = ax.twiny()
    ax.set_xticks(
        ticks=x,
        labels=s_section_labels,
    )
    # ax2.set_xticks(
    #     ticks=x,
    #     labels=s_section_labels,
    # )

    # ax.set_title("Accuracy vs LaREST Pipeline Stage")

    plt.savefig(
        output_dir / "accuracy.svg",
    )


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser: LarestArgumentParser = LarestArgumentParser()
    args: Namespace = parser.parse_args()

    # load LaREST config
    config: dict[str, Any] = get_config(args=args)

    # use style sheet
    sns.set_style("ticks")
    sns.set_context("paper")

    plot_accuracy(
        args=args,
        config=config,
        output_dir=Path("./assets"),
    )
