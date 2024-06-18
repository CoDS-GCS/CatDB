#!/bin/bash

dataset=$1
task_type=$2

# Run AutoML Baseline
#./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset AutoSklearn  30
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset H2O  30


#./explocal/exp3_end_to_end/runExperiment3_CAAFE.sh ${dataset} ${task_type} Yes TabPFN gpt-4
#./explocal/exp3_end_to_end/runExperiment3_CAAFE.sh ${dataset} ${task_type} No TabPFN gpt-3.5-turbo

#./explocal/exp3_end_to_end/runExperiment3_CAAFE.sh ${dataset} ${task_type} Yes RandomForest gpt-4
#./explocal/exp3_end_to_end/runExperiment3_CAAFE.sh ${dataset} ${task_type} No RandomForest gpt-4