#!/bin/bash

dataset=$1
task_type=$2

# Run CatDB Generated piplines
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} TEXT Random 0 1 ${task_type} gpt-4 
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} TEXT Random 0 1 ${task_type} gpt-3.5-turbo

# Run AutoML Framework
./explocal/exp2_micro_benchmark/automl_corresponding/${dataset}.sh ${task_type}
