#!/bin/bash

dataset=$1
log_file_name=$2

exp_path="$(pwd)"
data_source="${exp_path}/data"
data_profile_path="${exp_path}/setup/Baselines/kglids/kg_governor/data_profiling/src/"

conda activate kglids

SCRIPT="python main.py --data-source-name ${dataset} --data-source-path ${data_source} --output-path ${data_source}/${dataset}/data_profile/"

echo 3 >/proc/sys/vm/drop_caches && sync
sleep 3

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo ${dataset}",kglids,"$((($end - $start) / 1000000)) >>$log_file_name
