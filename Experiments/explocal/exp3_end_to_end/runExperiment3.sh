#!/bin/bash

dataset=$1
task_type=$2

# Run AutoML Baseline
iteration=1
deafult_max_run_time=1
CMD=./explocal/exp3_end_to_end/runExperiment3_AutoML.sh

$CMD $dataset AutoSklearn ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest 
$CMD $dataset H2O  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest 
$CMD $dataset Flaml  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest 
$CMD $dataset Autogluon ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest 


$CMD $dataset AutoSklearn ${deafult_max_run_time} ${iteration} llama3-70b-8192 
$CMD $dataset H2O  ${deafult_max_run_time} ${iteration} llama3-70b-8192 
$CMD $dataset Flaml  ${deafult_max_run_time} ${iteration} llama3-70b-8192 
$CMD $dataset Autogluon ${deafult_max_run_time} ${iteration} llama3-70b-8192 


$CMD $dataset AutoSklearn ${deafult_max_run_time} ${iteration} gpt-4o 
$CMD $dataset H2O  ${deafult_max_run_time} ${iteration} gpt-4o 
$CMD $dataset Flaml  ${deafult_max_run_time} ${iteration} gpt-4o 
$CMD $dataset Autogluon ${deafult_max_run_time} ${iteration} gpt-4o 
