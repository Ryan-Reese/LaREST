# LaREST: Lactone Ring-opening Energetics Sorting Tool

## Installation

1. **Conda**: The majority of the required packages for `LaREST` can be installed through the included [environment](./environment.yaml) file using `Conda`.

```bash
conda env create -f environment.yaml
conda activate larest
```

2. **CENSO**: `CENSO` will have to be seperately installed following the instructions on their [repository](https://github.com/grimme-lab/CENSO).

This can typically be done through cloning the `CENSO` repository and installing the package locally using `pip`. For example,

```bash
git clone https://github.com/grimme-lab/CENSO.git
pip install .
```

3. **ORCA**: `LaREST` (indirectly through `CENSO`) requires an `ORCA` installation to be available within the system's `PATH`.

The current release of `LaREST` has been tested with the larest release of `ORCA` (6.1.0).

> [!IMPORTANT]
> For different versions of `ORCA`, please remember to change the `orcaversion` [config](./config/config.toml) variable accordingly.


## Usage

`LaREST` has been written so that the entire computational pipeline can be customised within its [config](./config/config.toml) file.

For explanations/documentation for settings within `xTB`, `CREST`, or `CENSO`, please consult their doc pages. 


## Citations

For `xTB`:
- C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert, S. Spicher, S. Grimme
  *WIREs Comput. Mol. Sci.*, **2020**, 11, e01493.
  DOI: [10.1002/wcms.1493](https://doi.org/10.1002/wcms.1493)
- S. Grimme, C. Bannwarth, P. Shushkov,
  *J. Chem. Theory Comput.*, **2017**, 13, 1989-2009.
  DOI: [10.1021/acs.jctc.7b00118](https://dx.doi.org/10.1021/acs.jctc.7b00118)
- C. Bannwarth, S. Ehlert and S. Grimme.,
  *J. Chem. Theory Comput.*, **2019**, 15, 1652-1671.
  DOI: [10.1021/acs.jctc.8b01176](https://dx.doi.org/10.1021/acs.jctc.8b01176)
- P. Pracht, E. Caldeweyher, S. Ehlert, S. Grimme,
  *ChemRxiv*, **2019**, preprint.
  DOI: [10.26434/chemrxiv.8326202.v1](https://dx.doi.org/10.26434/chemrxiv.8326202.v1)
- S. Ehlert, M. Stahn, S. Spicher, S. Grimme,
  *J. Chem. Theory Comput.*, **2021**, 17, 4250-4261
  DOI: [10.1021/acs.jctc.1c00471](https://doi.org/10.1021/acs.jctc.1c00471)

For `CREST`:
 - P. Pracht, S. Grimme, C. Bannwarth, F. Bohle, S. Ehlert, G. Feldmann, J. Gorges, M. Müller, T. Neudecker, C. Plett, S. Spicher, P. Steinbach, P. Wesołowski, F. Zeller,
   *J. Chem. Phys.*, **2024**, *160*, 114110.
   DOI: [10.1063/5.0197592](https://doi.org/10.1063/5.0197592)
