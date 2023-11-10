
#!/bin/bash

# Systematic Experiments
dataset=$1
target_attribute=$2
log_file_name=$3

for time_left in 60 120 180 240 300 360 420 480 540 600
do

    for per_run_time_limit in 10 20 30 40 50 60
    do

        ./explocal/exp1_systematic/runExperiment1_AutoML_Classifier.sh  SKLearn_Classifier_AutoML ${dataset} ${target_attribute} ${time_left} ${per_run_time_limit} ${log_file_name}

        ./explocal/exp1_systematic/runExperiment1_AutoML_Classifier.sh  H2O_Classifier_AutoML ${dataset} ${target_attribute} ${time_left} ${per_run_time_limit} ${log_file_name}

    done

done