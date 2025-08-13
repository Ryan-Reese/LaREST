# LaREST: Lactone Ring-opening Energetics Sorting Tool

## Installation

1. **Conda**: The majority of the required packages for `LaREST` can be installed through the included [environment](./environment.yaml) file using `Conda`.

```bash
conda env create -f environment.yaml
conda activate larest
```

2. **CENSO**: `CENSO` will have to be seperately installed following the instructions on their [repository](https://github.com/grimme-lab/CENSO).

This can typically be done through cloning the `CENSO` repository and locally installing the package into our `Conda` environment.
For example,

```bash
git clone https://github.com/grimme-lab/CENSO.git
pip install .
```

> [!NOTE]
> The method of installation is actually unimportant, as long as the `CENSO` binary can be found in the system path.

3. **ORCA**: `LaREST` (indirectly through `CENSO`) requires an `ORCA` installation to be available within the system's `PATH`.

> [!IMPORTANT]
> For different versions of `ORCA`, please remember to change the `orcaversion` [config](./config/config.toml) variable accordingly.

4. **LaREST**: Lastly, we install `LaREST` as a package in our `Conda` environment.

```bash
git clone https://github.com/Ryan-Reese/LaREST.git
pip install .
```

## Usage

`LaREST` has been written so that the entire computational pipeline can be customised within its [config](./config/config.toml) file.

### For Imperial HPC Users

For users of `LaREST` via Imperial College's HPC service, an dedicates [pipeline script](./pipeline.sh) has been included.

Settings affecting job sizes can be changed by altering the first few lines of the job script.

```bash
#PBS -N LaREST
#PBS -l walltime=23:59:00
#PBS -l select=1:ncpus=128:mem=512gb:mpiprocs=128
```

Following the PBS directives, you will need to modify the following environment variables found in the `MODIFY ME` block:
- `CONDA_DIR`: **Please change this to the folder containing your `Conda` binary**
- `CONDA_ENV`: If the `Conda` environment was created using the [included file](./environment.yaml), you can leave this as `"larest"`
- `N_CORES`: The number of cores/MPI processes specified in the job script (default is `128`)

`LaREST` can then be run by navigating to the **parent `LaREST` directory** and submitting the job script using `qsub`. 
For instance,

```bash
cd LaREST.git
qsub pipeline.sh
```

## Dependencies

The current version of LaREST has been tested with the following:

Dependency | Version
--- | ---
`xTB` | 6.7.1 
`CREST` | 3.0.2 
`CENSO` | 2.1.4
`ORCA` | 6.1.0

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
