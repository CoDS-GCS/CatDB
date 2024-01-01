#!/bin/bash

dataset=$1
log_file_name=$2

exp_path="$(pwd)"
data_source="${exp_path}/data"
data_profile_path="${exp_path}/setup/Baselines/kglids/kg_governor/data_profiling/src/"

eval "$(conda shell.bash hook)"
conda activate kglids

cd ${data_profile_path}
rm -rf "${data_source}/${dataset}/data_profile/" #clean-up
SCRIPT="python main.py --data-source-name ${dataset}_train --data-source-path ${data_source}/${dataset} --output-path ${data_source}/${dataset}/data_profile/"

sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
sleep 3

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo ${dataset}",kglids,"$((($end - $start) / 1000000)) >>$log_file_name
