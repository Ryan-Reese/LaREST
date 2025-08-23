from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tropic.core.validate import MONOMER_GROUPS, get_func_group, get_ring_size

from larest.parsers import LarestArgumentParser
from larest.setup import get_config

PLOTTING_GROUPS = {
    "L": ["L"],
    "CC": ["CC"],
    "S-CC": ["CtC", "CdtC", "CtnC", "CX", "CtX"],
    "S-L": ["tL", "tnL", "dtL"],
    "Lm": ["Lm"],
}

COLORS = cycler(
    "color",
    [
        "#96559C",
        "#F4C15A",
        "#B5D67D",
        "#78C9F5",
        "#DC5E98",
        "#74E6B8",
        "#993f00",
        "#4c005c",
        "#426600",
        "#ff0010",
        "#9dcc00",
        "#c20088",
        "#003380",
        "#ffa405",
        "#ffff00",
        "#ff5005",
        "#5ef1f2",
        "#740aff",
        "#990000",
        "#00998f",
        "#005c31",
        "#2bce48",
        "#ffcc99",
        "#94ffb5",
        "#8f7c00",
        "#6fa8bb",
        "#808080",
    ],
).by_key()["color"]


def get_plotting_fg(fg):
    for group, fgs in PLOTTING_GROUPS.items():
        if fg in fgs:
            return group
    return None


def plot_predicted_vs_actual_enthalpy(
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
    # get computational results
    df: pd.DataFrame = pd.read_csv(
        Path(
            "./data",
            "crest_H.csv",
        ),
        index_col=None,
        comment="#",
    )

    df["functional_group"] = df["monomer_smiles"].apply(
        lambda x: get_func_group(x),
    )
    df = df.loc[df["functional_group"].notna()]

    df["functional_group"] = df["functional_group"].apply(
        lambda x: get_plotting_fg(x),
    )
    fg_codes, fg_uniques = df["functional_group"].factorize(sort=False)

    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot()
    ax.set_aspect("equal")

    # plot enthalpies
    scatter = ax.scatter(
        experimental_results["delta_H"],
        df["delta_H"],
        c=fg_codes,
        cmap="Accent",
        alpha=1.0,
        marker="x",
    )
    min_delta_h = min(
        *experimental_results["delta_H"],
        *df["delta_H"],
    )
    max_delta_h = max(
        *experimental_results["delta_H"],
        *df["delta_H"],
    )
    ax.set_xlim(min_delta_h - 20, max_delta_h + 10)
    ax.set_ylim(min_delta_h - 20, max_delta_h + 10)

    ax.set_xlabel(r"Exp $\Delta H_{p} \/ (\mathrm{kJ mol^{-1}})$")
    ax.set_ylabel(r"Comp $\Delta H_{p} \/ (\mathrm{kJ mol^{-1}})$")

    computed_values = df["delta_H"].notna()
    experimental_h = experimental_results[computed_values]["delta_H"]
    computational_h = df[computed_values]["delta_H"]

    mae = (experimental_h - computational_h).abs().mean()
    r2 = r2_score(experimental_h, computational_h)

    # determine best scalar adjustment
    difference_h = experimental_h - computational_h
    h_regressor = LinearRegression(fit_intercept=True)
    h_regressor.fit(
        np.zeros((difference_h.size, 1), dtype=float),
        difference_h.to_numpy(),
    )
    # plot both before and after lines
    ax.plot(
        [-200, 200],
        [-200, 200],
        color="tab:red",
        linestyle="dashed",
        alpha=0.5,
        linewidth=plt.rcParams["lines.linewidth"],
    )
    correction = h_regressor.intercept_
    ax.plot(
        [-200, 200],
        [-200 - correction, 200 - correction],
        color="tab:gray",
        linestyle="dashed",
        alpha=0.5,
        linewidth=plt.rcParams["lines.linewidth"],
    )
    # highlight key region
    ax.fill_between(
        [-20, 0],
        -200,
        200,
        alpha=0.1,
        color="tab:olive",
    )
    # add horizontal arrow
    midpoint = (min_delta_h + max_delta_h) / 2
    print(h_regressor.intercept_)
    ax.text(
        midpoint - (correction / 2) + 5,
        midpoint - (correction / 2),
        "Correction",
        ha="left",
        va="center",
        rotation=0,
        size=7,
        bbox=dict(
            boxstyle="rarrow,pad=0.3",
            fc="white",
            ec="tab:gray",
            lw=1,
        ),
    )
    handles, _ = scatter.legend_elements(alpha=0.6)
    ax.legend(
        handles,
        fg_uniques,
        loc="upper left",
        title="Functional Groups",
        ncols=1,
        # ncols=5,
        # columnspacing=plt.rcParams["legend.columnspacing"] / 2,
    )
    # ax.set_title(r"Experimental vs Computational $\Delta H$")

    plt.savefig(
        # output_dir / "predicted_vs_actual.svg",
        output_dir / "predicted_vs_actual.tif",
        dpi=600,
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

    plot_predicted_vs_actual_enthalpy(
        output_dir=Path("./assets"),
    )
