#!/usr/bin/bash
#PBS -N LaREST
#PBS -l walltime=23:59:00
#PBS -l select=1:ncpus=16:mem=128gb:mpiprocs=16

# source general config file
CONFIG_FILE="${PBS_O_WORKDIR}/config/general.conf"
if [ -e "${CONFIG_FILE}" ]; then
    source "${CONFIG_FILE}"
fi

# load orca
module load ORCA/6.1.0-gompi-2023b

# activate conda env
eval "$("${CONDA_DIR}"/conda shell.bash hook)"
conda activate ${CONDA_ENV}

# create run directory
DATETIME="$(date '+%Y%m%d%H%M%S')"
RUN_DIR="${PBS_O_WORKDIR}/${OUTPUT_DIR}/larest_${DATETIME}"
mkdir -p "${RUN_DIR}"

# copy config to run directory
cp -r "${PBS_O_WORKDIR}/${CONFIG_DIR}" "${RUN_DIR}"

# run LaREST
python "${PBS_O_WORKDIR}/${SRC_DIR}/main.py" -o "${RUN_DIR}" -c "${RUN_DIR}/${CONFIG_DIR}"

