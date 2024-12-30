#!/bin/bash

dataset=$1
task_type=$2

exp_path="$(pwd)"
log_file_name="${exp_path}/results/Experiment2_Data_Profile.dat"
data_source="${exp_path}/data"
metadata_source="${exp_path}/catalog"

mkdir -p ${metadata_source}
mkdir -p "${metadata_source}/${dataset}"
data_profile_path="${exp_path}/setup/Baselines/kglidsplus/kg_governor/data_profiling/src/"

eval "$(conda shell.bash hook)"
conda activate kglids

cd ${data_profile_path}
rm -rf "${metadata_source}/${dataset}/data_profile/" #clean-up
SCRIPT="python kglidsplus_main.py --data-source-name ${dataset} --data-source-path ${data_source}/${dataset} --output-path ${metadata_source}/${dataset}/data_profile/"

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo ${dataset}","${task_type}",kglidsplus,"$((($end - $start) / 1000000)) >>$log_file_name
