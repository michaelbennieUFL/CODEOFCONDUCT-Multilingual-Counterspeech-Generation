#!/bin/bash

source ./.venv/bin/activate # WRITE THE PATH TO YOUR ENVIROMENT

corpus=ML_MTCONAN_KN
export DEVICE="cuda:0" # GPU to use

export DATA_FOLD=./data/
export CORPUS=${corpus}
export GEN_FOLD=./generated/
export SAVE_FOLD=./evaluation/automatic_evaluation/

python ./evaluation/scripts/traditional_metrics.py