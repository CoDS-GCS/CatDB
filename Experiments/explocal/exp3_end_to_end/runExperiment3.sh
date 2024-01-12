#!/bin/bash

dataset=$1
task_type=$2
constraint="1h"

# Run AutoML Baseline
./explocal/exp1_systematic/runExperiment1_AutoML.sh $dataset $constraint