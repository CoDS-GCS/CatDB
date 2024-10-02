#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
target_attribute=$2

data_featurized_path="${exp_path}/data/${dataset}/data-featurized.csv"
data_input_down_path="${exp_path}/data/${dataset}/down-column-values.csv"

result_output_path="${exp_path}/results/Experiment1_Manual_DeduplicateData.dat"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python -Wignore main_manual_cleaning.py --root-data-path ${data_path} \
        --dataset-name ${dataset} \
        --result-output-path ${result_output_path} \
        --target-attribute ${target_attribute} \
        --data-featurized-path ${data_featurized_path} \
        --data-input-down-path ${data_input_down_path}"


echo ${SCRIPT}
$SCRIPT