#!/bin/bash

ROOT_DIR="/lustre06/project/6069023"
SCRATCH_DIR="/lustre07/scratch/jyaacoub/" # this is used for outputs on narval
BIN_DIR="${ROOT_DIR}/jyaacoub/bin" # for modeller

# Modeller is needed for this to run... (see: Generic install - https://salilab.org/modeller/10.5/release.html#unix)
export PYTHONPATH="${PYTHONPATH}:${BIN_DIR}/modeller10.5/lib/x86_64-intel8/python3.3:${BIN_DIR}/modeller10.5/lib/x86_64-intel8:${BIN_DIR}/modeller10.5/modlib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${BIN_DIR}/modeller10.5/lib/x86_64-intel8"

cd ${ROOT_DIR}/jyaacoub/MutDTA
source .venv/bin/activate

# export TRANSFORMERS_CACHE="${ROOT_DIR}/jyaacoub/hf_models/"
# export HF_HOME=${TRANSFORMERS_CACHE}
# export HF_HUB_OFFLINE=1

python -u playground.py