!#/usr/bin/bash

# WARN: this script should be run from larest directory

# TODO: confirm these job sizes

#PBS -N LaREST
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=16:mem=64gb

# source general config file
CONF_GENERAL="${PBS_O_WORKDIR}/config/general.conf"
if [ -e "${CONF_GENERAL}" ]; then
    source "${CONF_GENERAL}"
fi

# activate conda env
eval "$("${CONDA_DIR}"/conda shell.bash hook)"
conda activate ${CONDA_ENV}

# create output directories
RUN_DIR="${PBS_O_WORKDIR}/${OUTPUT_DIR}/larest_$(date '+%Y%m%d%H%M%S')"
mkdir -p "${RUN_DIR}/config/"
cp "${PBS_O_WORKDIR}/${CONFIG_DIR}/*" "${RUN_DIR}/${CONFIG_DIR}"
for i in {1..3}
do
    mkdir -p "${RUN_DIR}/step${i}/"
done

# run step 1 of pipeline
python "${PBS_O_WORKDIR}/${SRC_DIR}/step1.py"

# running crest
crest --input "${CONFIG_DIR}/crest.toml"
