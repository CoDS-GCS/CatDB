#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
source_dataset_name=$2
patch_src=$3
split_dataset=$4

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python -Wignore main_patch.py --dataset-name ${dataset} \
        --root-data-path ${data_path} \
        --source-dataset-name ${source_dataset_name} \
        --patch-src ${patch_src} \
        --split-dataset ${split_dataset}"


echo ${SCRIPT}

time $SCRIPT