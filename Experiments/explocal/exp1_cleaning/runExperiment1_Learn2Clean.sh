#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
task_type=$2

l2c_data_path="${data_path}/data_space/${dataset}"

result_output_path="${exp_path}/results/Experiment1_Learn2Clean.dat"

metadata_path="${l2c_data_path}/${dataset}_meta.csv"
test_data_path="${l2c_data_path}/${dataset}_orig_test.csv"
train_data_path="${l2c_data_path}/${dataset}_orig_train.csv"

cd "${exp_path}/setup/Baselines/Learn2Clean"
source venv/bin/activate

start=$(date +%s%N)
python -Wignore mainLearn2Clean.py --metadata-path "${l2c_data_path}/${dataset}.yaml" \
 --dataset-path "${data_path}/data_space" --result-output-path ${result_output_path} --test-data-path ${test_data_path} \
 --train-data-path ${train_data_path} --output-dir ${l2c_data_path}
end=$(date +%s%N)


# check L2C run sucessfully
# if [ ! -f "${l2c_data_path}/${dataset}_test.csv" ]; then
#     cp -r ${test_data_path} "${saga_data_path}/${dataset}_test.csv"
#     cp -r ${train_data_path} "${saga_data_path}/${dataset}_train.csv"
#     echo ${dataset} >> ${exp_path}/results/Experiment1_L2C_Fail.dat
# else
#     CMD="python -Wignore mainSAGARewriteConfig.py --metadata-path ${saga_data_path}/${dataset}.yaml --dataset-path ${data_path}/data_space"
#     $CMD   
# fi

