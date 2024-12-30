#!/bin/bash

dataset=$1
framework=$2
max_runtime_seconds=$3
iteration=$4
llm_model=$5
correspond_config=$6
jvm_memory=135

exp_path="$(pwd)"
data_path="${exp_path}/data/SAGA"
metadata_path="${data_path}/${dataset}/${dataset}.yaml"
output_dir="${exp_path}/results/AutoML"
output_path="${exp_path}/results/Experiment3_AutoML.dat"
exe_runtime_path="${exp_path}/archive/VLDB2025/results/AutoMLExeResults.csv"

mkdir -p ${output_dir}

cd "${exp_path}/setup/Baselines/AutoML"

CMD="python -Wignore main${framework}.py --metadata-path ${metadata_path} \
    --output-path ${output_path} \
    --max-runtime-seconds ${max_runtime_seconds} \
    --jvm-memory ${jvm_memory} \
    --dataset-path ${data_path} \
    --output-dir ${output_dir} \
    --iteration ${iteration} \
    --exe-runtime-path ${exe_runtime_path} \
    --llm-model ${llm_model}"

cd "${framework}AutoML"
source venv/bin/activate
cd ..

echo ${CMD}
$CMD