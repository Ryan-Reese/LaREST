from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from larest.constants import (
    HARTTREE_TO_JMOL,
    KCALMOL_TO_JMOL,
)


def parse_energy(output_file: Path) -> float:
    with open(output_file) as fstream:
        for line in fstream:
            if "FINAL SINGLE POINT ENERGY" in line:
                print(line)
                return float(line.split()[-1])
    return 0


def plot_conformers(output_dir: Path) -> None:
    # get experimental results
    result_files: list[Path] = [
        Path("rdkit_best.out"),
        Path("crest_best.out"),
        Path("censo_best.out"),
    ]
    image_files: list[Path] = [
        Path("./rdkit_best.png"),
        Path("./crest_best.png"),
        Path("./censo_best.png"),
    ]
    energies: pd.Series = pd.Series(
        [parse_energy(result_file) for result_file in result_files],
        dtype=np.float64,
    )
    energies *= HARTTREE_TO_JMOL / KCALMOL_TO_JMOL
    energies -= energies.iloc[0]

    colors = [
        "tab:green",
        "tab:blue",
        "tab:purple",
    ]
    section_names = [
        "RDKit",
        "CREST",
        # "CENSO Screening",
        # "CENSO Optimisation",
        # "CENSO Refinement",
        "CENSO",
    ]

    _, ax = plt.subplots(1, 1)

    for i, section_name in enumerate(section_names):
        ax.scatter(
            i,
            energies[i],
            s=(mpl.rcParams["lines.markersize"] ** 2) * 2,
            c=colors[i],
            marker="x",
            label=section_names[i],
        )
        ax.vlines(
            x=i,
            ymin=energies[i],
            ymax=0,
            color=colors[i],
            alpha=0.5,
            linestyle="dashed",
            linewidth=mpl.rcParams["lines.linewidth"],
        )
        # annotate vertical energy diff
        if i != 0:
            ax.annotate(
                f"{energies[i]:.2f}",
                xy=(i, float(energies[i])),
                xytext=(i - 0.25, energies[i] / 2),
                color="tab:red",
            )

    ax.axhline(
        y=0,
        color="tab:gray",
        alpha=0.5,
        linestyle="dashed",
        linewidth=mpl.rcParams["lines.linewidth"],
    )
    ax.set_ylabel(r"$\Delta E \/ (\mathrm{kcal mol^{-1}})$")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-14, 2)
    ax.set_xticks(
        ticks=np.arange(len(energies)),
        labels=section_names,
    )
    ax.legend(
        loc="best",
        title="LaREST Section",
        ncols=1,
    )
    # plt.show()

    plt.savefig(
        output_dir / "conformers.svg",
        # dpi=600,
    )


if __name__ == "__main__":
    # use style sheet
    sns.set_style("ticks")
    sns.set_context("paper")

    plot_conformers(
        output_dir=Path("./assets"),
    )
