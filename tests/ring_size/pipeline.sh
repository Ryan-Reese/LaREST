#!/usr/bin/bash
#PBS -N LaREST
#PBS -l walltime=23:59:00
#PBS -l select=1:ncpus=128:mem=512gb:mpiprocs=128

# Setting environment variables
OUTPUT_DIR="tests/ring_size/output"
CONFIG_DIR="tests/ring_size/config"
CONDA_DIR="${HOME}/miniforge3/bin"
CONDA_ENV="larest-main"
N_CORES=128

# xtb-required options
ulimit -s unlimited
ulimit -l unlimited
export OMP_STACKSIZE=4G
export OMP_NUM_THREADS="${N_CORES},1"
export OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1
export XTBPATH="$(pwd)"

# load orca
module load ORCA/6.1.0-gompi-2023b

# activate conda env
eval "$("${CONDA_DIR}"/conda shell.bash hook)"
conda activate ${CONDA_ENV}

# create run directory
mkdir -p "${PBS_O_WORKDIR}/${OUTPUT_DIR}"

# run LaREST
larest -o "${PBS_O_WORKDIR}/${OUTPUT_DIR}" -c "${PBS_O_WORKDIR}/${CONFIG_DIR}"
