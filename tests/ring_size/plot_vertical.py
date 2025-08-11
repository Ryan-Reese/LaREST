from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from larest.constants import ENTHALPY_PLOTTING_SECTIONS
from larest.parsers import LarestArgumentParser
from larest.setup import get_config


def plot_ring_size(
    args: Namespace,
    config: dict[str, Any],
    output_dir: Path,
):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    section_names = [
        "RDKit",
        "CREST",
        "CENSO Screening",
        "CENSO Optimisation",
        "CENSO Refinement",
    ]

    _, ax = plt.subplots(1, 1)

    # plot computational results
    for section_idx, section in enumerate(ENTHALPY_PLOTTING_SECTIONS):
        results_file: Path = Path(
            "./data",
            f"computational_{section}.csv",
        )

        results: pd.DataFrame = pd.read_csv(results_file)

        ax.plot(
            results["ring_size"],
            results["delta_H"],
            marker="s",
            linestyle="dashed",
            alpha=0.5,
            color=colors[section_idx],
            label=section_names[section_idx],
        )

    # plot experimental results
    experimental_results_file: Path = Path("./data/experimental.csv")
    experimental_results: pd.DataFrame = pd.read_csv(
        experimental_results_file,
        index_col=None,
        comment="#",
    )
    experimental_results = (
        experimental_results.groupby(
            "ring_size",
            sort=True,
            group_keys=True,
            dropna=False,
        )["delta_H"]
        .mean()
        .reset_index()
    )
    ax.plot(
        experimental_results["ring_size"],
        experimental_results["delta_H"],
        marker="x",
        linestyle="solid",
        alpha=1.0,
        color="black",
        label="experimental",
    )

    ax.set_ylabel(r"$\Delta H_{p} \/ (\mathrm{kJ mol^{-1}})$")
    ax.set_xlabel("Ring Size")
    ax.set_title(r"Effect of Ring Size for Monocyclic Unsubstituted Lactones")

    ax.legend(loc="best", ncols=1)

    plt.savefig(output_dir / "ring_size.svg")


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser: LarestArgumentParser = LarestArgumentParser()
    args: Namespace = parser.parse_args()

    # load LaREST config
    config: dict[str, Any] = get_config(args=args)

    # use style sheet
    sns.set_style("ticks")
    sns.set_context("paper")

    plot_ring_size(
        args=args,
        config=config,
        output_dir=Path("./assets"),
    )
