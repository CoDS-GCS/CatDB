#!/bin/bash

config=$1
dataset=$2
target_attribute=$3
time_left=$4
per_run_time_limit=$5
log_file_name=$6

SCRIPT="python3 setup/Baselines/${config}.py --dataset=data/${dataset}.csv --target_attribute=${target_attribute} --time_left=${time_left} --per_run_time_limit=${per_run_time_limit} --log_file_name=results/${log_file_name}.dat.acu"

echo $SCRIPT

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo ${config}","${dataset}","${time_left}","${per_run_time_limit}","$((($end - $start) / 1000000)) >>results/$log_file_name.dat
