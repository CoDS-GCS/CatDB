#!/bin/bash

dataset=$1

exp_path="$(pwd)"
data_source="${exp_path}/data"
data_profile_path="${exp_path}/setup/Baselines/kglids/kg_governor/data_profiling/src/"

eval "$(conda shell.bash hook)"
conda activate kglids

cd ${data_profile_path}
rm -rf "${data_source}/${dataset}/data_profile_full/" #clean-up
SCRIPT="python kglids_main.py --data-source-name ${dataset} --data-source-path ${data_source}/${dataset} --output-path ${data_source}/${dataset}/data_profile_full/"

$SCRIPT