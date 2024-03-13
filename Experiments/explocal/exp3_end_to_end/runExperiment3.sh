#!/bin/bash

dataset=$1
# task_type=$2
# constraint="1h"

# root_path="$(pwd)"
# benchmark_path="${root_path}/setup/automlbenchmark/resources/"
# config_path="${root_path}/setup/config/automl/constraints.yaml"

# cp -r ${config_path} ${benchmark_path} 

# # Run AutoML Baseline
# ./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset $constraint


./explocal/exp3_end_to_end/runExperiment3_CAFFE.sh $dataset