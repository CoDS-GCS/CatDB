#!/bin/bash

dataset=$1
task_type=$2

# Run AutoML Baseline
iteration=1
deafult_max_run_time=1

./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset AutoSklearn ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDBChain
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset H2O  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDBChain
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset Flaml  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDBChain
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset Autogluon ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDBChain


./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset AutoSklearn ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDBChain
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset H2O  ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDBChain
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset Flaml  ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDBChain
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset Autogluon ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDBChain


./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset AutoSklearn ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDB
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset H2O  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDB
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset Flaml  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDB
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset Autogluon ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest CatDB


./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset AutoSklearn ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDB
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset H2O  ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDB
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset Flaml  ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDB
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset Autogluon ${deafult_max_run_time} ${iteration} llama3-70b-8192 CatDB