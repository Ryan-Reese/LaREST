from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from larest.base import Monomer
from larest.constants import ENTHALPY_PLOTTING_SECTIONS, ENTROPY_PLOTTING_SECTIONS
from larest.parsers import LarestArgumentParser
from larest.setup import get_config


def plot_enthalpy(
    args: Namespace,
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
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    section_names = [
        "RDKit",
        "CREST",
        "CENSO Screening",
        "CENSO Optimisation",
        "CENSO Refinement",
    ]
    monomer_lengths = config["reaction"]["lengths"]

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

            best_fit_x = np.linspace(0, x[-1], num=2)
            best_fit_y = h_regressor.predict(best_fit_x.reshape(-1, 1))

            monomer_name = experimental_results.index[monomer_idx]
            experimental_h = experimental_results.iloc[monomer_idx]["delta_H"]
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
                alpha=0.5,
            )
            # plot computational intercept
            ax.scatter(
                0,
                h_regressor.intercept_,
                s=(2 * (mpl.rcParams["lines.markersize"]) ** 2),
                marker="x",
                color=colors[section_idx],
                label=section_names[section_idx],
            )

            ax.set_ylabel(r"$\Delta H_{p} \/ (\mathrm{kJ mol^{-1}})$")
            ax.set_xlabel(r"$1/L$")
            ax.set_ylim(-100, 0)
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
                ticks=([0] + [1 / length for length in monomer_lengths[::-1]]),
                labels=(
                    [r"$\infty$"] + [str(length) for length in monomer_lengths[::-1]]
                ),
            )
            ax.set_xlim(-0.05, (1 / monomer_lengths[0]) + 0.05)
            ax2.set_xlim(-0.05, (1 / monomer_lengths[0]) + 0.05)
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


def plot_entropy(
    args: Namespace,
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
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    section_names = [
        "RDKit",
        "CREST",
        "CREST w/ correction",
        "CENSO Refinement",
        "CENSO Refinement w/ correction",
    ]
    monomer_lengths = config["reaction"]["lengths"]

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

        # iterating over sections
        for section_idx, section in enumerate(ENTROPY_PLOTTING_SECTIONS):
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

            best_fit_x = np.linspace(0, x[-1], num=2)
            best_fit_y = s_regressor.predict(best_fit_x.reshape(-1, 1))

            monomer_name = experimental_results.index[monomer_idx]
            experimental_s = experimental_results.iloc[monomer_idx]["delta_S"]
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
                alpha=0.5,
            )
            # plot computational intercept
            ax.scatter(
                0,
                s_regressor.intercept_,
                s=(2 * (mpl.rcParams["lines.markersize"]) ** 2),
                marker="x",
                color=colors[section_idx],
                label=section_names[section_idx],
            )

            ax.set_ylabel(r"$\Delta S_{p} \/ (\mathrm{J mol^{-1}})$")
            ax.set_xlabel(r"$1/L$")
            ax.set_ylim(-200, 0)
            ax.vlines(
                0,
                -200,
                0,
                colors="tab:gray",
                linestyles="dashed",
                alpha=0.5,
            )
            ax2 = ax.twiny()
            ax2.set_xticks(
                ticks=([0] + [1 / length for length in monomer_lengths[::-1]]),
                labels=(
                    [r"$\infty$"] + [str(length) for length in monomer_lengths[::-1]]
                ),
            )
            ax.set_xlim(-0.05, (1 / monomer_lengths[0]) + 0.05)
            ax2.set_xlim(-0.05, (1 / monomer_lengths[0]) + 0.05)
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
                output_dir / f"{monomer_name}_S.svg",
            )


def plot_accuracy(
    args: Namespace,
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

    h_absolute_errors: list[list[float]] = [
        [] for _ in range(len(ENTHALPY_PLOTTING_SECTIONS))
    ]
    s_absolute_errors: list[list[float]] = [
        [] for _ in range(len(ENTROPY_PLOTTING_SECTIONS))
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
            h_absolute_errors[section_idx].append(abs(predicted_h - experimental_h))

        # iterating over sections
        for section_idx, section in enumerate(ENTROPY_PLOTTING_SECTIONS):
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

            predicted_s = s_regressor.intercept_
            experimental_s = experimental_results.iloc[monomer_idx]["delta_S"]
            s_absolute_errors[section_idx].append(abs(predicted_s - experimental_s))

    _, ax = plt.subplots(1, 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    h_section_labels = [
        "RDKit",
        "CREST",
        "Screening",
        "Optimisation",
        "Refinement",
    ]
    s_section_labels = [
        "RDKit",
        "CREST",
        "CREST w/ correction",
        "CENSO",
        "CENSO w/ correction",
    ]

    h_mean_absolute_errors: list[float] = [
        sum(errors) / len(errors) for errors in h_absolute_errors
    ]
    h_std_absolute_errors: list[float] = [
        np.std(errors).item() for errors in h_absolute_errors
    ]
    s_mean_absolute_errors: list[float] = [
        sum(errors) / len(errors) for errors in s_absolute_errors
    ]
    s_std_absolute_errors: list[float] = [
        np.std(errors).item() for errors in s_absolute_errors
    ]

    # plot MAEs
    x = np.arange(len(ENTHALPY_PLOTTING_SECTIONS))
    ax.errorbar(
        x,
        h_mean_absolute_errors,
        yerr=h_std_absolute_errors,
        ecolor="tab:gray",
        elinewidth=(0.8 * mpl.rcParams["lines.linewidth"]),
        capsize=4.0,
        color=colors[0],
        linestyle="dashed",
        linewidth=(1.2 * mpl.rcParams["lines.linewidth"]),
        marker="s",
        markersize=(1.2 * mpl.rcParams["lines.markersize"]),
    )
    ax2 = ax.twinx()
    ax2.errorbar(
        x,
        s_mean_absolute_errors,
        yerr=s_std_absolute_errors,
        ecolor="tab:gray",
        elinewidth=(0.8 * mpl.rcParams["lines.linewidth"]),
        capsize=4.0,
        color=colors[1],
        linestyle="dashed",
        linewidth=(1.2 * mpl.rcParams["lines.linewidth"]),
        marker="s",
        markersize=(1.2 * mpl.rcParams["lines.markersize"]),
    )

    ax.set_ylabel(r"$\Delta H$ MAE $(\mathrm{kJ mol^{-1}})$")
    ax2.set_ylabel(r"$\Delta S$ MAE $(\mathrm{J mol^{-1}})$")
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    ax2 = ax.twiny()
    ax.set_xticks(
        ticks=x,
        labels=h_section_labels,
    )
    ax2.set_xticks(
        ticks=x,
        labels=s_section_labels,
    )

    ax.set_title("Accuracy vs LaREST Pipeline Stage")

    plt.savefig(
        output_dir / "accuracy.svg",
    )


def plot_predicted_vs_actual(
    args: Namespace,
    config: dict[str, Any],
    output_dir: Path,
):
    # get experimental results
    experimental_results: pd.DataFrame = pd.read_csv(
        Path("./experimental.csv"),
        index_col="name",
        comment="#",
    )
    # get computational results
    computational_results: pd.DataFrame = pd.read_csv(
        Path("./computational.csv"),
        index_col="name",
        comment="#",
    )

    _, ax = plt.subplots(1, 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot enthalpies
    ax.scatter(
        experimental_results["delta_H"],
        computational_results["delta_H"],
        color="black",
        marker="x",
        label="Post-CENSO",
    )
    ax.plot(
        [-100, 100],
        [-100, 100],
        color="tab:red",
        linestyle="dashed",
        label="Tu et al.",
    )
    min_delta_h = min(
        *experimental_results["delta_H"],
        *computational_results["delta_H"],
    )
    max_delta_h = max(
        *experimental_results["delta_H"],
        *computational_results["delta_H"],
    )
    ax.set_xlim(min_delta_h - 5, max_delta_h + 5)
    ax.set_ylim(min_delta_h - 5, max_delta_h + 5)

    ax.set_xlabel(r"$\Delta H_{exp} (\mathrm{kJ mol^{-1}})$")
    ax.set_ylabel(r"$\Delta H_{comp} (\mathrm{kJ mol^{-1}})$")

    computed_values = computational_results["delta_H"].notna()
    experimental_h = experimental_results["delta_H"][computed_values]
    computational_h = computational_results["delta_H"][computed_values]
    absolute_errors = (experimental_h - computational_h).abs()

    mae = absolute_errors.mean()
    h_regressor = LinearRegression(fit_intercept=True)
    h_regressor.fit(
        computational_h.to_numpy().reshape(-1, 1),
        experimental_h.to_numpy(),
    )

    r2 = r2_score(
        experimental_results["delta_H"][computed_values],
        computational_results["delta_H"][computed_values],
    )
    ax.text(
        0.01,
        0.99,
        f"MAE: {mae}\nR2: {h_regressor.score(computational_h.to_numpy().reshape(-1, 1), experimental_h.to_numpy())}",
        style="italic",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox={"pad": 10},
    )
    ax.legend(loc="best")
    ax.set_title(r"Experimental vs Computational $\Delta H$")

    plt.savefig(
        output_dir / "predicted_vs_actual.svg",
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

    # plot_accuracy(
    #     args=args,
    #     config=config,
    #     output_dir=Path("./assets"),
    # )

    plot_predicted_vs_actual(
        args=args,
        config=config,
        output_dir=Path("./assets"),
    )
