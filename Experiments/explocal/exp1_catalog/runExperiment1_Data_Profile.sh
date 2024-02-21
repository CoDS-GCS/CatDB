#!/bin/bash

dataset=$1
task_type=$2

exp_path="$(pwd)"
log_file_name="${exp_path}/results/Experiment1_Data_Profile.dat"
data_source="${exp_path}/data"
data_profile_path="${exp_path}/setup/Baselines/kglids/kg_governor/data_profiling/src/"

eval "$(conda shell.bash hook)"
conda activate kglids

cd ${data_profile_path}
rm -rf "${data_source}/${dataset}/data_profile/" #clean-up
SCRIPT="python kglids_main.py --data-source-name ${dataset}_train --data-source-path ${data_source}/${dataset} --output-path ${data_source}/${dataset}/data_profile/"

# sudo sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
# sleep 3

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo ${dataset}","${task_type}",kglids,"$((($end - $start) / 1000000)) >>$log_file_name
