#!/bin/bash

exp_path="$(pwd)"
dataset=$1
task_type=$2
data_path="${exp_path}/data"
metadata_path="${data_path}/${dataset}/${dataset}.yaml"

number_iteration=2
log_file_name_execution="${exp_path}/results/Experiment3_CAAFE.dat"
log_file_name_resultst="${exp_path}/results/Experiment3_CAAFE_Results.dat"
log_file_name_nohup="${exp_path}/results/Experiment3_CAAFE_nohup_${dataset}.dat"
log_file_name_error="${exp_path}/results/Experiment3_CAAFE_ERROR_${dataset}.dat"

cd "${exp_path}/setup/Baselines/CAAFE"
source venv/bin/activate

SCRIPT="python CAAFEV2.py --metadata-path ${metadata_path} --log-file-name ${log_file_name_resultst}  --number-iteration ${number_iteration} > ${log_file_name_nohup} 2>${log_file_name_error} < /dev/null"

echo "${SCRIPT}"

# sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
# sleep 3

start=$(date +%s%N)
bash -c "nohup ${SCRIPT}" 
end=$(date +%s%N)
echo ${dataset}",CAFFE,"${task_type}","$((($end - $start) / 1000000)) >>$log_file_name_execution

cd ${exp_path}