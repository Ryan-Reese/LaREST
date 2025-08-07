import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles

DATASHEET: str = "./input_experimental.xlsx"


def format_for_functional_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df["standardised_smiles"] = df["monomer_smiles"].apply(
        lambda x: StandardizeSmiles(x)
    )
    df = df.drop_duplicates("standardised_smiles")
    df["functional_group"] = df["standardised_smiles"].apply(
        lambda x: get_func_group(x)
    )
    df = df.groupby("functional_group").count()
    df = df.drop(columns=["monomer_smiles"])
    df = df.rename({"standardised_smiles": "total"})

    return df


def format_for_publications(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates("doi").dropna()
    df["date"] = df["date"].astype(int)
    df = df.groupby("date").count().reset_index()
    df["total"] = df["doi"].cumsum()
    return df

    # plt.show()
    plt.savefig("./assets/publications.svg")


def get_experimental_data() -> None:
    df: pd.DataFrame = pd.read_excel(
        DATASHEET,
        sheet_name=0,
        header=0,
        usecols=[
            "monomer_smiles",
            "medium",
            "monomer_state",
            "polymer_state",
            "temperature",
            "delta_h",
            "delta_s",
            "date",  # used to determine if entry is valid
        ],
    )
    # remove calorimetry values for temperatures other than 298.15K
    df = df.loc[df["temperature"].isna() | (df["temperature"].isin([298, 298.15]))]
    df = df.drop(columns=["temperature"])

    # remove entries with missing data
    df = df.dropna()

    # standardise monomer smiles strings
    df["monomer_smiles"] = df["monomer_smiles"].apply(
        lambda x: StandardizeSmiles(x),
    )

    # used to compare with larest results
    df_larest = (
        df.groupby(
            "monomer_smiles",
            sort=True,
            group_keys=True,
        )[["delta_h", "delta_s"]]
        .mean()
        .reset_index()
    )
    # df = df.sort_values("standardised_smiles")
    df_larest.to_csv("./experimental.csv", index=False, header=True)

    # contains secondary information for analysis
    df.sort_values(["monomer_smiles", "date"]).to_csv(
        "./experimental_detailed.csv",
        index=False,
        header=True,
    )


if __name__ == "__main__":
    get_experimental_data()
