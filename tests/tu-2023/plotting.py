import argparse
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from larest.base import Monomer
from larest.constants import ENTHALPY_PLOTTING_SECTIONS
from larest.parsers import LarestArgumentParser
from larest.setup import get_config

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


def plot_extrapolation(
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
):
    # get experimental results
    experimental_results_file: Path = Path("./experimental.csv")
    experimental_results: pd.DataFrame = pd.read_csv(
        experimental_results_file,
        index_col="name",
        comment="#",
    )

    # iterate over monomers
    for monomer_idx, monomer_smiles in enumerate(config["reaction"]["monomers"]):
        monomer: Monomer = Monomer(
            smiles=monomer_smiles,
            args=args,
            config=config,
        )

        summary_dir: Path = Path(monomer.dir_path, "summary")

        # check if results exist
        if not summary_dir.exists():
            continue

        _, ax = plt.subplots(1, 1)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        section_names = [
            "RDKit",
            "CREST",
            "CENSO Screening",
            "CENSO Optimisation",
            "CENSO Refinement",
        ]

        # iterating over sections
        for section_idx, section in enumerate(ENTHALPY_PLOTTING_SECTIONS):
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

            best_fit_x = np.linspace(0, x[-1], num=50)
            best_fit_y = h_regressor.predict(best_fit_x.reshape(-1, 1))

            monomer_name = experimental_results.index[monomer_idx]
            experimental_h = experimental_results.iloc[monomer_idx]["delta_H"]
            # plot experimental result
            ax.scatter(
                0,
                experimental_h,
                s=(2 * (mpl.rcParams["lines.markersize"]) ** 2),
                c="black",
                marker="x",
                label="Experimental",
            )
            ax.annotate(
                r"Exp $\Delta H_{p}$",
                xy=(0, experimental_h),
                xytext=(0.10, experimental_h),
                arrowprops={"facecolor": "black", "shrink": 0.05},
            )

            # plot computational results
            ax.scatter(
                x,
                y,
                color=colors[section_idx],
                marker="o",
                label=section_names[section_idx],
            )
            # plot computational LOBF
            ax.plot(
                best_fit_x,
                best_fit_y,
                linestyle="dashed",
                label=section_names[section_idx],
                color=colors[section_idx],
                # alpha=0.5,
            )
            # plot computational intercept
            ax.scatter(
                0,
                h_regressor.intercept_,
                s=(2 * (mpl.rcParams["lines.markersize"]) ** 2),
                marker="o",
                color=colors[section_idx],
                label=section_names[section_idx],
            )

            ax.set_ylabel(r"$\Delta H_{p} \/ (\mathrm{kJ mol^{-1}})$")
            ax.set_xlabel(r"$1/L$")
            ax.set_xlim(-0.05, 0.55)
            ax.vlines(
                0,
                -100,
                0,
                colors="tab:gray",
                linestyles="dashed",
                alpha=0.5,
            )
            ax2 = ax.twiny()
            ax2.set_xticks(
                ticks=[0, 1 / 4, 1 / 3, 1 / 2],
                labels=[r"$\infty$", "4", "3", "2"],
            )
            ax2.set_xlim(-0.05, 0.55)
            ax2.set_xlabel(r"Polymer Length ($L$)")

            # ax.set_xlim(right=1.25)

            handles, labels = ax.get_legend_handles_labels()
            unique_handles_labels = [
                (handle, label)
                for i, (handle, label) in enumerate(zip(handles, labels, strict=True))
                if label not in labels[:i]
            ]

            ax.legend(
                *zip(*unique_handles_labels, strict=True),
                loc="lower right",
                ncols=1,
            )

            # ax.set_title(str(monomer_name))

            plt.savefig(
                output_dir / f"{monomer_name}_H.svg",
            )


def plot_accuracy(
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
):
    # get experimental results
    experimental_results_file: Path = Path("./experimental.csv")
    experimental_results: pd.DataFrame = pd.read_csv(
        experimental_results_file,
        index_col="name",
        comment="#",
    )

    absolute_errors: list[list[float]] = [
        [] for _ in range(len(ENTHALPY_PLOTTING_SECTIONS))
    ]

    # iterate over monomers
    for monomer_idx, monomer_smiles in enumerate(config["reaction"]["monomers"]):
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

            x = (1 / results["polymer_length"]).to_numpy()[::-1]
            y = (results["delta_H"] / 1000).to_numpy()[::-1]

            h_regressor = LinearRegression(fit_intercept=True)
            h_regressor.fit(x.reshape(-1, 1), y)

            predicted_h = h_regressor.intercept_
            experimental_h = experimental_results.iloc[monomer_idx]["delta_H"]
            absolute_errors[section_idx].append(abs(predicted_h - experimental_h))

    _, ax = plt.subplots(1, 1)

    section_labels = [
        "RDKit",
        "CREST",
        "Screening",
        "Optimisation",
        "Refinement",
    ]

    mean_absolute_errors: list[float] = [
        sum(errors) / len(errors) for errors in absolute_errors
    ]
    std_absolute_errors: list[float] = [
        np.std(errors).item() for errors in absolute_errors
    ]

    # plot MAEs
    x = np.arange(len(mean_absolute_errors))
    ax.errorbar(
        x,
        mean_absolute_errors,
        yerr=std_absolute_errors,
        ecolor="tab:gray",
        elinewidth=(0.8 * mpl.rcParams["lines.linewidth"]),
        capsize=4.0,
        color="tab:gray",
        linestyle="dashed",
        linewidth=(1.2 * mpl.rcParams["lines.linewidth"]),
        marker="s",
        markersize=(1.2 * mpl.rcParams["lines.markersize"]),
    )
    # ax.hlines(
    #     0,
    #     x[0],
    #     x[-1],
    #     colors="tab:gray",
    #     linestyles="dashed",
    # )

    ax.set_ylabel(r"Mean Absolute Error $(\mathrm{kJ mol^{-1}})$")
    ax.set_xlabel("LaREST Pipeline Stage")
    ax.set_ylim(bottom=0)
    ax.set_xticks(
        ticks=x,
        labels=section_labels,
    )
    # ax2 = ax.twiny()
    # ax2.set_xlim(-0.05, 0.55)
    # ax2.set_xlabel(r"Polymer Length ($L$)")

    # ax.set_xlim(right=1.25)

    # handles, labels = ax.get_legend_handles_labels()
    # unique_handles_labels = [
    #     (handle, label)
    #     for i, (handle, label) in enumerate(zip(handles, labels, strict=True))
    #     if label not in labels[:i]
    # ]
    #
    # ax.legend(
    #     *zip(*unique_handles_labels, strict=True),
    #     loc="lower right",
    #     ncols=1,
    # )
    #
    # ax.set_title(str(monomer_name))

    plt.savefig(
        output_dir / "accuracy_H.svg",
    )


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser: LarestArgumentParser = LarestArgumentParser()
    args: argparse.Namespace = parser.parse_args()

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

    # plot_extrapolation(
    #     args=args,
    #     config=config,
    #     output_dir=Path("./assets", "extrapolation"),
    # )
