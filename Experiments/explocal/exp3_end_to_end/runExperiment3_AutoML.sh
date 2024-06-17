#!/bin/bash

dataset=$1
max_runtime_seconds=$2
jvm_memory=20

exp_path="$(pwd)"
data_path="${exp_path}/data"
metadata_path="${data_path}/${dataset}/${dataset}.yaml"
output_dir="${exp_path}/results/AutoML"
output_path="${exp_path}/results/Experiment3_AutoML.dat"

mkdir -p ${output_dir}

cd "${exp_path}/setup/Baselines/AutoML"

source venv/bin/activate

CMD="python -Wignore main.py --metadata-path ${metadata_path} \
    --output-path ${output_path} \
    --max-runtime-seconds ${max_runtime_seconds} \
    --jvm-memory ${jvm_memory} \
    --dataset-path ${exp_path} \
    --output-dir ${output_dir}"

# $CMD --automl-framework H2O
# $CMD --automl-framework FLAML
$CMD --automl-framework Autogluon

cd ${exp_path}