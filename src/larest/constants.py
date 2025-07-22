HARTTREE_TO_JMOL = 2625499.63948
KCALMOL_TO_JMOL = 4184.0
CALMOL_TO_JMOL = 4.184
MONOMER_GROUPS: dict[str, str] = {
    "CC": "[O;R]-[C;R](=[O;!R])-[O;R]",
    "CtC": "[O;R]-[C;R](=[O;!R])-[S;R]",
    "CdtC": "[S;R]-[C;R](=[O;!R])-[S;R]",
    "CtnC": "[O;R]-[C;R](=[S;!R])-[O;R]",
    "CX": "[O;R]-[C;R](=[S;!R])-[S;R]",
    "CtX": "[S;R]-[C;R](=[S;!R])-[S;R]",
    "L": "[C,c;R]-[C;R](=[O;!R])-[O;R]",
    "tL": "[C,c;R]-[C;R](=[O;!R])-[S;R]",
    "tnL": "[C,c;R]-[C;R](=[S;!R])-[O;R]",
    "dtL": "[C,c;R]-[C;R](=[S;!R])-[S;R]",
    "oA": "[O;R]-[C;R](=[O;!R])-[N;R]",
    "Lm": "[C,c;R]-[C;R](=[O;!R])-[N;R]",
}
INITIATOR_GROUPS: dict[str, str] = {"OH": "C[OH]"}
XTB_OUTPUT_PARAMS: list[str] = [
    "H",
    "S",
    "G",
    "U",
]
CREST_OUTPUT_PARAMS: list[str] = [
    "S_conf",
    "S_rrho",
    "S_total",
]
CENSO_OUTPUT_PARAMS: list[str] = [
    "G",
    "H",  # internal energy taken as enthalpy
    "S",
]
CENSO_SECTIONS: list[str] = [
    "0_PRESCREENING",
    "1_SCREENING",
    "2_OPTIMIZATION",
    "3_REFINEMENT",
]
PIPELINE_SECTIONS: list[str] = [
    "rdkit",
    "crest",
    "0_PRESCREENING",
    "1_SCREENING",
    "2_OPTIMIZATION",
    "3_REFINEMENT",
]
