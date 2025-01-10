#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
task_type=$2

mkdir -p "${data_path}/data_space"
mkdir -p "${data_path}/data_space/${dataset}"

l2c_data_path="${data_path}/${dataset}"
result_output_path="${exp_path}/results/Experiment1_Learn2Clean.dat"

metadata_path="${l2c_data_path}/${dataset}.yaml"
cp -r ${metadata_path} "${data_path}/data_space/${dataset}/"

cd "${exp_path}/setup/Baselines/Learn2Clean"
source venv/bin/activate

start=$(date +%s%N)

python -Wignore mainLearn2Clean.py --metadata-path ${metadata_path} --dataset-path ${data_path} \
    --result-output-path ${result_output_path} --output-dir "${data_path}/data_space/${dataset}"

end=$(date +%s%N)