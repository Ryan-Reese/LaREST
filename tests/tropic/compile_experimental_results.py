import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles

DATASHEET: str = "./input_experimental.xlsx"


def get_experimental_data() -> None:
    df: pd.DataFrame = pd.read_excel(
        DATASHEET,
        sheet_name=0,
        header=0,
        usecols=[
            "monomer_smiles",
            "polymerisation_type",
            "medium",
            "solvent",
            "monomer_state",
            "polymer_state",
            "temperature",
            "delta_h",
            "delta_s",
            "doi",
            "flag",  # used to determine if entry is valid
        ],
    )
    # remove calorimetry values for temperatures other than 298.15K
    df = df.loc[df["temperature"].isna() | (df["temperature"].isin([298, 298.15]))]
    df = df.drop(columns=["temperature"])

    # remove flagged entries
    df = df.loc[df["flag"].isna()]
    df = df.drop(columns=["flag"])

    # remove entries with missing delta_h or delta_s
    df = df.loc[df["delta_h"].notna() & df["delta_s"].notna()]

    # standardise and sort monomer smiles strings
    df["monomer_smiles"] = df["monomer_smiles"].apply(
        lambda x: StandardizeSmiles(x),
    )
    df = df.sort_values(["monomer_smiles"])

    # extract only delta_h and delta_s for LaREST
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
    df_larest.to_csv(
        "./experimental.csv",
        index=False,
        header=True,
    )

    # contains secondary information for analysis
    df.to_csv(
        "./experimental_detailed.csv",
        index=False,
        header=True,
    )


if __name__ == "__main__":
    get_experimental_data()
