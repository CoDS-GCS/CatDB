
#!/bin/bash

# Systematic Experiments
dataset=$1
target_attribute=$2
log_file_name=$3

# run AutoML 

./explocal/exp1_systematic/runExperiment1_AutoML.sh  

# for time_left in {240..240}; #240 300
# do

#     for per_run_time_limit in {30..30}; #10 20 30
#     do

#         ./explocal/exp1_systematic/runExperiment1_AutoML_Classifier.sh  SKLearn_Classifier_AutoML ${dataset} ${target_attribute} ${time_left} ${per_run_time_limit} ${log_file_name}

#         #./explocal/exp1_systematic/runExperiment1_AutoML_Classifier.sh  H2O_Classifier_AutoML ${dataset} ${target_attribute} ${time_left} ${per_run_time_limit} ${log_file_name}

#     done

# done

#./explocal/exp1_systematic/runExperiment1_AutoML_Classifier.sh  ChatGPT_KDD99 ${dataset} ${target_attribute} 0 0 ${log_file_name}

