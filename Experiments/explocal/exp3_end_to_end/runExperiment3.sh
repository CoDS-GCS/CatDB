#!/bin/bash

dataset=$1
task_type=$2
constraint="1h"

# Run AutoML Baseline
./explocal/exp3_end_to_end/runExperiment3_AutoML.sh $dataset $constraint