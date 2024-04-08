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
# echo "dataset,iteration,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,task_type,time" >> "${exp_path}/results/Experiment1_LLM_Pipe_Test.dat"

echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,task_type,time,result" >> "${exp_path}/results/Experiment2_CatDB_LLM_Pipe_Run.dat"
# echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment2_AutoML_Corresponding.dat"

# echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment3_AutoML_1H.dat"
# echo "dataset,source,time" >> "${exp_path}/results/Experiment3_Hand-craft.dat"

cd ${exp_path}

#CMD=./explocal/exp0_statistics/runExperiment0.sh
CMD=./explocal/exp1_catalog/runExperiment1.sh
#CMD=./explocal/exp2_micro_benchmark/runExperiment2.sh 
#CMD=./explocal/exp3_end_to_end/runExperiment3.sh 


$CMD Higgs binary
# $CMD albert binary
# $CMD Click_prediction_small binary
# $CMD Census-Augmented binary
# $CMD BNG_heart-statlog binary
# $CMD KDDCup99_full multiclass
# $CMD road-safety multiclass
# $CMD drug-directory multiclass
# $CMD okcupid-stem multiclass
# $CMD walking-activity multiclass
# $CMD PASS regression
# $CMD aloi regression
# $CMD MD_MIX_Mini_Copy regression
# $CMD dionis regression
# $CMD Meta_Album_BRD_Extended regression


# $CMD balance-scale multiclass
# $CMD breast-w binary
# $CMD cmc multiclass
# $CMD credit-g binary
# $CMD diabetes binary
# $CMD tic-tac-toe binary
# $CMD eucalyptus multiclass
# $CMD pc1 binary
# $CMD airlines binary
# $CMD jungle_chess_2pcs_raw_endgame_complete multiclass
