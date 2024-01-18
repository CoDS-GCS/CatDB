#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;
mkdir -p catdb-results;

exp_path="$(pwd)"

# Add headers to log files
echo "dataset,platform,time" >> "${exp_path}/results/Experiment1_Data_Profile.dat"
echo "dataset,time" >> "${exp_path}/results/Experiment1_CSVDataReader.dat"
echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,prompt_number_iteration,task_type,time,result" >> "${exp_path}/results/Experiment1_LLM_Pipe_Gen.dat"

echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,prompt_number_iteration,task_type,time,result" >> "${exp_path}/results/Experiment2_CatDB_LLM_Pipe_Run.dat"
echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment2_AutoML_Corresponding.dat"

echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment3_AutoML_1H.dat"

cd ${exp_path}

CMD=./explocal/exp1_catalog/runExperiment1.sh
#CMD=./explocal/exp2_micro_benchmark/runExperiment2.sh 

$CMD simulated_electricity binary 
$CMD KDD98 binary 
$CMD Higgs binary 
$CMD airlines binary 
$CMD BNG_credit_g binary 
$CMD Microsoft multiclass 
$CMD cmc multiclass 
$CMD diabetes multiclass 
$CMD 3-million-Sudoku-puzzles-with-ratings multiclass 
$CMD pokerhand multiclass 
$CMD Buzzinsocialmedia_Twitter regression 
$CMD delays_zurich_transport regression 
$CMD nyc-taxi-green-dec-2016 regression 
$CMD black_friday regression 
$CMD federal_election regression 


## Large Datasets 
## Binary Datasets
# $CMD IMDB.drama binary 
# $CMD 20_newsgroups.drift binary 
# $CMD Epsilon binary 
# $CMD prostate binary 
# $CMD bates_classif_100 binary 
# $CMD simulated_electricity binary 
# $CMD sf-police-incidents binary 
# $CMD KDDCup09-Appetency binary 
# $CMD Higgs binary 

# # Multiclass Datasets
# $CMD STL-10 multiclass 
# $CMD Microsoft multiclass 
# $CMD CIFAR-100 multiclass 
# $CMD GTSRB-HOG03 multiclass 
# $CMD BNG_solar_flare multiclass 
# $CMD CovPokElec muliclass 
# $CMD beer_reviews multiclass 
# $CMD Traffic_violations muliclass 

# # Regression Datasets
# $CMD Buzzinsocialmedia_Twitter regression 
# $CMD Tallo regression 
# $CMD NYC regression 
# $CMD USCars regression 
# $CMD CanadaPricePrediction regression 