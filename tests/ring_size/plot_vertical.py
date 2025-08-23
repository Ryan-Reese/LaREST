from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from larest.parsers import LarestArgumentParser
from larest.setup import get_config


def plot_ring_size_enthalpy(
    output_dir: Path,
):
    _, ax = plt.subplots(1, 1)

    # get computational results
    results_file: Path = Path(
        "./data",
        "3_REFINEMENT_H.csv",
    )
    results: pd.DataFrame = pd.read_csv(results_file)

    # plot computational results
    ax.plot(
        results["ring_size"],
        results["delta_H"],
        marker="x",
        linestyle="dashed",
        alpha=1,
        color="tab:purple",
        label="LaREST",
        linewidth=mpl.rcParams["lines.linewidth"],
    )

    # plot experimental results
    experimental_results_file: Path = Path(
        "./data",
        "experimental.csv",
    )
    experimental_results: pd.DataFrame = pd.read_csv(
        experimental_results_file,
        index_col=None,
        comment="#",
    )
    ax.plot(
        experimental_results["ring_size"],
        experimental_results["delta_H"],
        marker="s",
        linestyle="solid",
        alpha=0.5,
        color="tab:gray",
        label="TROPIC Average",
    )

    # plot lebedev results
    lebedev_results_file: Path = Path(
        "./data",
        "lebedev.csv",
    )
    lebedev_results: pd.DataFrame = pd.read_csv(
        lebedev_results_file,
        index_col=None,
        comment="#",
    )
    ax.plot(
        lebedev_results["ring_size"],
        lebedev_results["delta_H"],
        marker="^",
        linestyle="solid",
        alpha=0.5,
        color="black",
        label="Lebedev (1996)",
    )

    # fill boundary
    has_h = results["ring_size"].notna()
    ax.fill_between(
        results["ring_size"][has_h],
        results["delta_H"][has_h],
        lebedev_results[lebedev_results["ring_size"].isin(results["ring_size"][has_h])][
            "delta_H"
        ],
        color="tab:red",
        alpha=0.1,
    )

    ax.set_ylabel(r"$\Delta H_{p} \/ (\mathrm{kJ mol^{-1}})$")
    ax.set_xlabel(r"Monomer Ring Size ($N$)")
    # ax.set_title(r"Effect of Ring Size for Monocyclic Unsubstituted Lactones")

    ax.legend(loc="best", ncols=1)

    plt.savefig(
        # output_dir / "ring_size_H.svg",
        output_dir / "ring_size_H.tif",
        dpi=600,
    )


def plot_ring_size_entropy(
    args: Namespace,
    config: dict[str, Any],
    output_dir: Path,
):
    colors = ["tab:purple", "tab:pink"]
    labels = [
        "LaREST",
        r"LaREST w/ $S_{conf}$",
    ]

    _, ax = plt.subplots(1, 1)

    # plot computational results

    for section_idx, section_name in enumerate(
        [
            "3_REFINEMENT",
            "censo_corrected",
        ],
    ):
        results_file: Path = Path(
            "./data",
            f"{section_name}_S.csv",
        )

        results: pd.DataFrame = pd.read_csv(results_file)

        ax.plot(
            results["ring_size"],
            results["delta_S"],
            marker="x",
            linestyle="dashed",
            alpha=1,
            color=colors[section_idx],
            label=labels[section_idx],
            linewidth=mpl.rcParams["lines.linewidth"],
        )

    # plot experimental results
    experimental_results_file: Path = Path(
        "./data",
        "experimental.csv",
    )
    experimental_results: pd.DataFrame = pd.read_csv(
        experimental_results_file,
        index_col=None,
        comment="#",
    )
    ax.plot(
        experimental_results["ring_size"],
        experimental_results["delta_S"],
        marker="s",
        linestyle="solid",
        alpha=0.5,
        color="tab:gray",
        label="TROPIC Average",
    )

    # plot lebedev results
    lebedev_results_file: Path = Path(
        "./data",
        "lebedev.csv",
    )
    lebedev_results: pd.DataFrame = pd.read_csv(
        lebedev_results_file,
        index_col=None,
        comment="#",
    )
    ax.plot(
        lebedev_results["ring_size"],
        lebedev_results["delta_S"],
        marker="^",
        linestyle="solid",
        alpha=0.5,
        color="black",
        label="Lebedev (1996)",
    )

    # fill boundary
    has_s = results["ring_size"].notna()
    ax.fill_between(
        results["ring_size"][has_s],
        results["delta_S"][has_s],
        lebedev_results[lebedev_results["ring_size"].isin(results["ring_size"][has_s])][
            "delta_S"
        ],
        color="tab:red",
        alpha=0.1,
    )

    ax.set_ylabel(r"$\Delta S_{p} \/ (\mathrm{J mol^{-1} K^{-1}})$")
    ax.set_xlabel(r"Monomer Ring Size ($N$)")
    # ax.set_title(r"Effect of Ring Size for Monocyclic Unsubstituted Lactones")

    ax.legend(loc="best", ncols=1)

    plt.savefig(output_dir / "ring_size_S.svg")


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser: LarestArgumentParser = LarestArgumentParser()
    args: Namespace = parser.parse_args()

    # load LaREST config
    config: dict[str, Any] = get_config(args=args)

    # use style sheet
    sns.set_style("ticks")
    sns.set_context("paper")

    plot_ring_size_enthalpy(
        output_dir=Path("./assets"),
    )

    plot_ring_size_entropy(
        args=args,
        config=config,
        output_dir=Path("./assets"),
    )
