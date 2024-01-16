#!/bin/bash

dataset=$1

exp_path="$(pwd)"
log_file_name="${exp_path}/results/Experiment1_CSVDataReader.dat"
data_source="${exp_path}/data"

cd "${exp_path}/setup/config/"
source venv/bin/activate

SCRIPT="python CSVDataReader.py ${data_source} ${dataset}"

# sudo sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
# sleep 3

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo ${dataset}","$((($end - $start) / 1000000)) >>$log_file_name
