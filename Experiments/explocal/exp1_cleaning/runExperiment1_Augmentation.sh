#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1

# clean-up 
output_dir="${data_path}/Augmentation"
mkdir -p $output_dir

adasyn_data_path="${output_dir}/${dataset}"
mkdir -p $adasyn_data_path

cp -r "${data_path}/${dataset}/${dataset}.yaml" $adasyn_data_path
cp -r "${data_path}/${dataset}/${dataset}_test.csv" $adasyn_data_path

metadata_path="${adasyn_data_path}/${dataset}.yaml"

cd "${exp_path}/setup/Baselines/Augmentation"
source venv/bin/activate

CMD="python -Wignore mainAugmentation.py --metadata-path ${metadata_path} \
        --output-dir ${output_dir} --dataset-path ${data_path}"

start=$(date +%s%N)
$CMD
end=$(date +%s%N)

log_file_name="${exp_path}/results/Experiment1_Augmentation.dat"
echo ${dataset}","$((($end - $start) / 1000000)) >>$log_file_name