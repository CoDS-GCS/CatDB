#!/bin/bash

dataset=$1
out_path=$2

exp_path="$(pwd)"
data_source="${exp_path}/data"
data_profile_path="${exp_path}/setup/Baselines/kglids/kg_governor/data_profiling/src/"

eval "$(conda shell.bash hook)"
conda activate kglids

cd ${data_profile_path}
rm -rf ${out_path} #clean-up
SCRIPT="python kglids_main.py --data-source-name ${dataset} --data-source-path ${data_source}/${dataset} --output-path ${out_path}"

$SCRIPT