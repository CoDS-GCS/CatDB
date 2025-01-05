#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1

# clean-up 
output_dir="${data_path}/data_space"
aug_data_path="${output_dir}/${dataset}"
metadata_path="${aug_data_path}/${dataset}.yaml"

cd "${exp_path}/setup/Baselines/Augmentation"
source venv/bin/activate

CMD="python -Wignore mainAugmentation.py --metadata-path ${metadata_path} \
        --output-dir ${output_dir} --dataset-path ${output_dir}"

start=$(date +%s%N)
$CMD
end=$(date +%s%N)

log_file_name="${exp_path}/results/Experiment1_Augmentation.dat"
echo ${dataset}","$((($end - $start) / 1000000)) >>$log_file_name