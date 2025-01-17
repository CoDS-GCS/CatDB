#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
task_type=$2

mkdir -p "${data_path}/data_space"
mkdir -p "${data_path}/data_space/${dataset}"

l2c_data_path="${data_path}/data_space/${dataset}"
result_output_path="${exp_path}/results/Experiment1_Learn2Clean.dat"

cp -r "${data_path}/${dataset}/${dataset}.yaml" "${data_path}/data_space/${dataset}/${dataset}_Learn2Clean.yaml"
metadata_path="${data_path}/data_space/${dataset}/${dataset}_Learn2Clean.yaml"

sed -i 's/_train.csv/_orig_train.csv/g' $metadata_path
sed -i 's/_test.csv/_orig_test.csv/g' $metadata_path

cd "${exp_path}/setup/Baselines/Learn2Clean"
source venv/bin/activate

start=$(date +%s%N)

python -Wignore mainLearn2Clean.py --metadata-path ${metadata_path} --dataset-path "${data_path}/data_space" \
    --result-output-path ${result_output_path} --output-dir "${l2c_data_path}"

end=$(date +%s%N)