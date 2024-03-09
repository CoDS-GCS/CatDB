#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;
mkdir -p catdb-results;

exp_path="$(pwd)"

# Add headers to log files
echo "dataset,task_type,platform,time" >> "${exp_path}/results/Experiment1_Data_Profile.dat"
echo "dataset,time" >> "${exp_path}/results/Experiment1_CSVDataReader.dat"
echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,task_type,time" >> "${exp_path}/results/Experiment1_LLM_Pipe_Gen.dat"
echo "dataset,iteration,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,task_type,time" >> "${exp_path}/results/Experiment1_LLM_Pipe_Test.dat"

echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,task_type,time,result" >> "${exp_path}/results/Experiment2_CatDB_LLM_Pipe_Run.dat"
echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment2_AutoML_Corresponding.dat"

echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment3_AutoML_1H.dat"
echo "dataset,source,time" >> "${exp_path}/results/Experiment3_Hand-craft.dat"

cd ${exp_path}

#CMD=./explocal/exp0_statistics/runExperiment0.sh
CMD=./explocal/exp1_catalog/runExperiment1.sh
#CMD=./explocal/exp2_micro_benchmark/runExperiment2.sh 
#CMD=./explocal/exp3_end_to_end/runExperiment3.sh 


#$CMD dataset_1_rnc binary test
$CMD dataset_2_rnc binary test
# $CMD dataset_3_rnc binary test
# $CMD dataset_4_rnc binary test
# $CMD dataset_5_rnc binary test
# $CMD dataset_6_rnc binary test

# $CMD oml_dataset_1_rnc binary test
# $CMD oml_dataset_2_rnc binary test
# $CMD oml_dataset_3_rnc binary test
# $CMD oml_dataset_4_rnc binary test
# $CMD oml_dataset_5_rnc multiclass test
# $CMD oml_dataset_6_rnc multiclass test
# $CMD oml_dataset_7_rnc regression test
# $CMD oml_dataset_8_rnc regression test
# $CMD oml_dataset_9_rnc regression test
