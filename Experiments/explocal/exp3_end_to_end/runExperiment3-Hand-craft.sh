#!/bin/bash


root_path="$(pwd)"
baselines_path="${root_path}/baselines"
log_file_name="${root_path}/results/Experiment3_Hand-craft.dat"

cd ${baselines_path}

source venv/bin/activate


start=$(date +%s%N)
python diabetes_kaggle.py
end=$(date +%s%N)

echo "diabetes,kaggle-gold-hand-craft,"$((($end - $start) / 1000000)) >>$log_file_name

