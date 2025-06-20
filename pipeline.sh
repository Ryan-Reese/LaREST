!#/usr/bin/bash

# WARN: this script should be run from larest

# TODO: confirm these job sizes

#PBS -N LaREST
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=16:mem=128gb

# source general config file
CONFIG_FILE="${PBS_O_WORKDIR}/config/general.conf"
if [ -e "${CONFIG_FILE}" ]; then
    source "${CONFIG_FILE}"
fi

# activate conda env
eval "$("${CONDA_DIR}"/conda shell.bash hook)"
conda activate ${CONDA_ENV}

# create run directory
DATETIME="$(date '+%Y%m%d%H%M%S')"
RUN_DIR="${PBS_O_WORKDIR}/${OUTPUT_DIR}/larest_${DATETIME}"
mkdir -p "${RUN_DIR}/${CONFIG_DIR}"

# copy config to run directory
cp "${PBS_O_WORKDIR}/${CONFIG_DIR}/*" "${RUN_DIR}/${CONFIG_DIR}"

# run step 1 of pipeline (xTB)
python "${PBS_O_WORKDIR}/${SRC_DIR}/step1.py" -o "${RUN_DIR}" -c "${RUN_DIR}/${CONFIG_DIR}"

# run step 2 of pipeline (CREST)
# python "${PBS_O_WORKDIR}/${SRC_DIR}/step2.py" -o "${RUN_DIR}" -c "${RUN_DIR}/${CONFIG_DIR}"
