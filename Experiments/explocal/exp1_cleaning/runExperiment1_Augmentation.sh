#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
source_data=$2

# clean-up 
output_dir="${data_path}/data_space"
aug_data_path="${output_dir}/${dataset}"

metadata_path="${aug_data_path}/${dataset}_${source_data}.yaml"

if [[ $source_data == "orig" ]]
    then
        metadata_path="${aug_data_path}/${dataset}.yaml"
        if [ ! -f "${output_dir}/${dataset}/${dataset}_orig_train.csv" ]; then
                cp -r "${data_path}/${dataset}/${dataset}_train.csv" "${output_dir}/${dataset}/${dataset}_orig_train.csv"
        fi     

        if [ ! -f "${output_dir}/${dataset}/${dataset}.yaml" ]; then
                cp -r "${data_path}/${dataset}/${dataset}.yaml" "${output_dir}/${dataset}/${dataset}.yaml"
        fi            
fi

cd "${exp_path}/setup/Baselines/Augmentation"
source venv/bin/activate

CMD="python -Wignore mainAugmentation.py --metadata-path ${metadata_path} \
        --output-dir ${output_dir} --dataset-path ${output_dir} --source-data ${source_data}"

start=$(date +%s%N)
$CMD
end=$(date +%s%N)

log_file_name="${exp_path}/results/Experiment1_Augmentation.dat"
echo ${dataset}","$source_data","$((($end - $start) / 1000000)) >>$log_file_name