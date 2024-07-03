#!/bin/bash

dataset=$1
task_type=$2

# Run AutoML Baseline
iteration=1
deafult_max_run_time=1
CMD=./explocal/exp3_end_to_end/runExperiment3_AutoML.sh

$CMD $dataset AutoSklearn ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDBChain
$CMD $dataset H2O  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDBChain
$CMD $dataset Flaml  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDBChain
$CMD $dataset Autogluon ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDBChain


$CMD $dataset AutoSklearn ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDBChain
$CMD $dataset H2O  ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDBChain
$CMD $dataset Flaml  ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDBChain
$CMD $dataset Autogluon ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDBChain


$CMD $dataset AutoSklearn ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDB
$CMD $dataset H2O  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDB
$CMD $dataset Flaml  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDB
$CMD $dataset Autogluon ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDB


$CMD $dataset AutoSklearn ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDB
$CMD $dataset H2O  ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDB
$CMD $dataset Flaml  ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDB
$CMD $dataset Autogluon ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDB