#!/bin/bash

dataset=$1


exp_path="$(pwd)"
log_file_name="${exp_path}/results/Experiment3_CAAFE.dat"
log_file_name_dataset="${exp_path}/results/Experiment3_CAAFE_${dataset}.dat"
error_file_name="${exp_path}/results/Experiment3_CAAFE_ERROR_${dataset}.dat"

cd "${exp_path}/setup/Baselines/CAAFE"
source venv/bin/activate

SCRIPT1="python scripts/generate_features_script.py --dataset_id=${dataset} --prompt_id=v4> ${log_file_name_dataset} 2>${error_file_name} < /dev/null"
SCRIPT="python scripts/run_classifiers_script.py --dataset_id=${dataset} --prompt_id=v4> ${log_file_name_dataset} 2>${error_file_name} < /dev/null"

echo "${SCRIPT}"

# sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
# sleep 3

start=$(date +%s%N)
bash -c "nohup ${SCRIPT}" 
end=$(date +%s%N)
echo ${dataset}","CAFFE","$((($end - $start) / 1000000)) >>$log_file_name

cd ${exp_path}