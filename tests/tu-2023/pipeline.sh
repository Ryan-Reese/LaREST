#!/usr/bin/bash
#PBS -N LaREST
#PBS -l walltime=23:59:00
#PBS -l select=1:ncpus=16:mem=128gb:mpiprocs=16

# Setting environment variables
SRC_DIR="src/larest"
OUTPUT_DIR="tests/tu-2023/output"
CONFIG_DIR="tests/tu-2023/config"
CONDA_DIR="${HOME}/miniforge3/bin"
CONDA_ENV="larest"
N_CORES=16

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

# DATETIME="$(date '+%Y%m%d%H%M%S')"
# RUN_DIR="${PBS_O_WORKDIR}/${OUTPUT_DIR}/larest_${DATETIME}"

# create run directories
mkdir -p "${PBS_O_WORKDIR}/${OUTPUT_DIR}"
mkdir -p "${PBS_O_WORKDIR}/${CONFIG_DIR}"

# copy config to run directory
# cp -r "${PBS_O_WORKDIR}/${CONFIG_DIR}" "${RUN_DIR}"

# run LaREST
python "${PBS_O_WORKDIR}/${SRC_DIR}/main.py" -o "${PBS_O_WORKDIR}/${OUTPUT_DIR}" -c "${PBS_O_WORKDIR}/${CONFIG_DIR}"
