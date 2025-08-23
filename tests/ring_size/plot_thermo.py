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


def plot_enthalpy(
    args: Namespace,
    config: dict[str, Any],
    output_dir: Path,
):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    labels = [
        "RDKit",
        "CREST",
        "CENSO",
    ]
    monomer_lengths = config["reaction"]["lengths"]

    # get experimental results
    experimental_results_file: Path = Path(
        "./data",
        "tropic.csv",
    )
    experimental_results: pd.DataFrame = pd.read_csv(
        experimental_results_file,
        index_col=None,
        comment="#",
    )

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

        _, ax = plt.subplots(1, 1)

        # save values for setting margins
        all_enthalpies: list[float] = []

        # iterating over sections
        for section_idx, section_name in enumerate(ENTHALPY_PLOTTING_SECTIONS):
            results_file: Path = summary_dir / f"{section_name}.csv"

            results: pd.DataFrame = pd.read_csv(
                results_file,
                header=0,
                dtype=np.float64,
            )

            if results["delta_H"].hasnans:
                continue

            # compute LaREST result
            x = (1 / results["polymer_length"]).to_numpy()[::-1]
            y = (results["delta_H"] / 1000).to_numpy()[::-1]

            h_regressor = LinearRegression(fit_intercept=True)
            h_regressor.fit(x.reshape(-1, 1), y)

            print(f"{monomer_smiles=} {section_name=}")
            print(h_regressor.score(x.reshape(-1, 1), y))

            best_fit_x = np.linspace(0, x[-1], num=2)
            best_fit_y = h_regressor.predict(best_fit_x.reshape(-1, 1))

            # plot computational results
            ax.scatter(
                x,
                y,
                color=colors[section_idx],
                marker="o",
            )
            # plot computational LOBF
            ax.plot(
                best_fit_x,
                best_fit_y,
                linestyle="dashed",
                label=labels[section_idx],
                color=colors[section_idx],
                alpha=0.5,
            )
            # plot computational intercept
            ax.scatter(
                0,
                h_regressor.intercept_,
                s=(2 * (mpl.rcParams["lines.markersize"]) ** 2),
                marker="x",
                color=colors[section_idx],
            )
            # label computational intercept
            if section_idx == (len(ENTHALPY_PLOTTING_SECTIONS) - 1):
                ax.annotate(
                    r"$ \lim_{N\to\infty} \Delta H_{p} $",
                    xy=(0, h_regressor.intercept_),
                    xytext=(0.10, h_regressor.intercept_),
                    bbox=dict(
                        boxstyle="round4,pad=1.0",
                        facecolor="white",
                        edgecolor="black",
                    ),
                    arrowprops={
                        "facecolor": "black",
                        "shrink": 0.05,
                    },
                )
            all_enthalpies += list(y)
            all_enthalpies.append(h_regressor.intercept_)

        try:
            # get experimental result
            experimental_h = experimental_results.loc[
                experimental_results["monomer_smiles"]
                == StandardizeSmiles(monomer_smiles)
            ]["delta_H"].item()

            # plot experimental result
            ax.scatter(
                0,
                experimental_h,
                s=(2 * (mpl.rcParams["lines.markersize"] ** 2)),
                c="black",
                marker="x",
                label="Experimental",
            )
            ax.annotate(
                r"Exp $\Delta H_{p}$",
                xy=(0, experimental_h),
                xytext=(0.10, experimental_h),
                bbox=dict(
                    boxstyle="round4,pad=1.0",
                    facecolor="white",
                    edgecolor="black",
                ),
                arrowprops={
                    "facecolor": "black",
                    "shrink": 0.05,
                },
            )
            all_enthalpies.append(experimental_h)
        except ValueError:
            pass

        ax.set_ylabel(r"$\Delta H_{p,N} \/ (\mathrm{kJ mol^{-1}})$")
        ax.set_xlabel(r"$1/N$")
        ax.set_ylim(
            min(all_enthalpies) - 20,
            max(all_enthalpies) + 20,
        )
        ax.vlines(
            0,
            -400,
            400,
            colors="tab:gray",
            linestyles="dashed",
            alpha=0.5,
        )
        ax2 = ax.twiny()
        ax2.set_xticks(
            ticks=([0] + [1 / length for length in monomer_lengths[::-1]]),
            labels=([r"$\infty$"] + [str(length) for length in monomer_lengths[::-1]]),
        )
        ax.set_xlim(-0.05, (1 / monomer_lengths[0]) + 0.05)
        ax2.set_xlim(-0.05, (1 / monomer_lengths[0]) + 0.05)
        ax2.set_xlabel(r"Polymer Length ($N$)")

        handles, labels = ax.get_legend_handles_labels()
        unique_handles_labels = [
            (handle, label)
            for i, (handle, label) in enumerate(zip(handles, labels, strict=True))
            if label not in labels[:i]
        ]

        ax.legend(
            *zip(*unique_handles_labels, strict=True),
            loc="lower right",
            title="LaREST Section",
            ncols=1,
        )

        # ax.set_title(str(monomer_name))

        plt.savefig(
            output_dir / f"{monomer.smiles}_H.svg",
            # dpi=600,
        )


def plot_entropy(
    args: Namespace,
    config: dict[str, Any],
    output_dir: Path,
):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    labels = [
        "RDKit",
        "CREST",
        "CENSO",
        r"CENSO w/ $S_{conf}$",
    ]
    polymer_lengths = config["reaction"]["lengths"]

    # get experimental results
    experimental_results_file: Path = Path(
        "./data",
        "tropic.csv",
    )
    experimental_results: pd.DataFrame = pd.read_csv(
        experimental_results_file,
        index_col=None,
        comment="#",
    )

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

        _, ax = plt.subplots(1, 1)

        # save values for setting margins
        all_entropies: list[float] = []

        # iterating over sections
        for section_idx, section_name in enumerate(ENTROPY_PLOTTING_SECTIONS):
            results_file: Path = summary_dir / f"{section_name}.csv"

            results: pd.DataFrame = pd.read_csv(
                results_file,
                header=0,
                dtype=np.float64,
            )

            # check if results fully exist
            if results["delta_S"].hasnans:
                continue

            # compute LaREST result
            x = (1 / results["polymer_length"]).to_numpy()[::-1]
            y = (results["delta_S"]).to_numpy()[::-1]

            s_regressor = LinearRegression(fit_intercept=True)
            s_regressor.fit(x.reshape(-1, 1), y)

            best_fit_x = np.linspace(0, x[-1], num=2)
            best_fit_y = s_regressor.predict(best_fit_x.reshape(-1, 1))

            # plot computational results
            ax.scatter(
                x,
                y,
                color=colors[section_idx],
                marker="o",
            )
            # plot computational LOBF
            ax.plot(
                best_fit_x,
                best_fit_y,
                linestyle="dashed",
                label=labels[section_idx],
                color=colors[section_idx],
                alpha=0.5,
            )
            # plot computational intercept
            ax.scatter(
                0,
                s_regressor.intercept_,
                s=(2 * (mpl.rcParams["lines.markersize"]) ** 2),
                marker="x",
                color=colors[section_idx],
            )
            # label computational intercept
            if section_idx == (len(ENTROPY_PLOTTING_SECTIONS) - 1):
                ax.annotate(
                    r"$ \lim_{N\to\infty} \Delta S_{p} $",
                    xy=(0, s_regressor.intercept_),
                    xytext=(0.10, s_regressor.intercept_),
                    bbox=dict(
                        boxstyle="round4,pad=1.0",
                        facecolor="white",
                        edgecolor="black",
                    ),
                    arrowprops={
                        "facecolor": "black",
                        "shrink": 0.05,
                    },
                )
            all_entropies += list(y)
            all_entropies.append(s_regressor.intercept_)

        try:
            # get experimental result
            experimental_s = experimental_results.loc[
                experimental_results["monomer_smiles"]
                == StandardizeSmiles(monomer_smiles)
            ]["delta_S"].item()

            # plot experimental result
            ax.scatter(
                0,
                experimental_s,
                s=(2 * (mpl.rcParams["lines.markersize"] ** 2)),
                c="black",
                marker="x",
                label="Experimental",
            )
            ax.annotate(
                r"Exp $\Delta S_{p}$",
                xy=(0, experimental_s),
                xytext=(0.10, experimental_s),
                bbox=dict(
                    boxstyle="round4,pad=1.0",
                    facecolor="white",
                    edgecolor="black",
                ),
                arrowprops={
                    "facecolor": "black",
                    "shrink": 0.05,
                },
            )
            all_entropies.append(experimental_h)
        except ValueError:
            pass

        ax.set_ylabel(r"$\Delta S_{p,N} \/ (\mathrm{J mol^{-1} K^{-1}})$")
        ax.set_xlabel(r"$1/N$")
        ax.set_ylim(
            min(all_entropies) - 20,
            max(all_entropies) + 20,
        )
        ax.vlines(
            0,
            -400,
            400,
            colors="tab:gray",
            linestyles="dashed",
            alpha=0.5,
        )
        ax2 = ax.twiny()
        ax2.set_xticks(
            ticks=([0] + [1 / length for length in polymer_lengths[::-1]]),
            labels=([r"$\infty$"] + [str(length) for length in polymer_lengths[::-1]]),
        )
        ax.set_xlim(-0.05, (1 / polymer_lengths[0]) + 0.05)
        ax2.set_xlim(-0.05, (1 / polymer_lengths[0]) + 0.05)
        ax2.set_xlabel(r"Polymer Length ($N$)")

        handles, labels = ax.get_legend_handles_labels()
        unique_handles_labels = [
            (handle, label)
            for i, (handle, label) in enumerate(zip(handles, labels, strict=True))
            if label not in labels[:i]
        ]

        ax.legend(
            *zip(*unique_handles_labels, strict=True),
            loc="lower right",
            title="LaREST Section",
            ncols=1,
        )

        # ax.set_title(str(monomer_name))

        plt.savefig(
            output_dir / f"{monomer.smiles}_S.svg",
            # dpi=600,
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

    plot_enthalpy(
        args=args,
        config=config,
        output_dir=Path("./assets", "extrapolation"),
    )

    # plot_entropy(
    #     args=args,
    #     config=config,
    #     output_dir=Path("./assets", "extrapolation"),
    # )
