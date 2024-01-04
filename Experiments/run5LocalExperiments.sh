#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;
mkdir -p llm-results;

exp_path="$(pwd)"

# Add headers to log files
echo "dataset,platform,time" >> "${exp_path}/results/Experiment1_Data_Profile.dat"
echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment1_AutoML.dat"
echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,prompt_number_iteration,task_type,time" >> "${exp_path}/results/Experiment1_CatDB.dat"

./explocal/exp1_systematic/runExperiment1.sh airlines binary 
# ./explocal/exp1_systematic/runExperiment1.sh albert binary 
# ./explocal/exp1_systematic/runExperiment1.sh covertype multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh dionis multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh adult binary 
# ./explocal/exp1_systematic/runExperiment1.sh Amazon_employee_access binary 
# ./explocal/exp1_systematic/runExperiment1.sh APSFailure binary 
# ./explocal/exp1_systematic/runExperiment1.sh bank-marketing binary 
# ./explocal/exp1_systematic/runExperiment1.sh connect-4 multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh Fashion-MNIST multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh guillermo binary 
# ./explocal/exp1_systematic/runExperiment1.sh helena multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh higgs binary 
# ./explocal/exp1_systematic/runExperiment1.sh jannis multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh jungle_chess_2pcs_raw_endgame_complete multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh KDDCup09_appetency binary 
# ./explocal/exp1_systematic/runExperiment1.sh MiniBooNE binary 
# ./explocal/exp1_systematic/runExperiment1.sh nomao binary 
# ./explocal/exp1_systematic/runExperiment1.sh numerai28.6 binary 
# ./explocal/exp1_systematic/runExperiment1.sh riccardo binary 
# ./explocal/exp1_systematic/runExperiment1.sh robert multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh shuttle multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh volkert multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh Australian binary 
# ./explocal/exp1_systematic/runExperiment1.sh blood-transfusion-service-center binary 
# ./explocal/exp1_systematic/runExperiment1.sh car multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh christine binary 
# ./explocal/exp1_systematic/runExperiment1.sh cnae-9 multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh credit-g binary 
# ./explocal/exp1_systematic/runExperiment1.sh dilbert multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh fabert multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh jasmine binary 
# ./explocal/exp1_systematic/runExperiment1.sh kc1 binary 
# ./explocal/exp1_systematic/runExperiment1.sh kr-vs-kp binary 
# ./explocal/exp1_systematic/runExperiment1.sh mfeat-factors multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh phoneme binary 
# ./explocal/exp1_systematic/runExperiment1.sh segment multiclass 
# ./explocal/exp1_systematic/runExperiment1.sh sylvine binary 
# ./explocal/exp1_systematic/runExperiment1.sh vehicle multiclass 

#Binary classification datasets:
###############################
# ./explocal/exp1_systematic/runExperiment1.sh dorothea BinaryClassification
# ./explocal/exp1_systematic/runExperiment1.sh christine BinaryClassification
# ./explocal/exp1_systematic/runExperiment1.sh jasmine BinaryClassification
# ./explocal/exp1_systematic/runExperiment1.sh philippine BinaryClassification
# ./explocal/exp1_systematic/runExperiment1.sh madeline BinaryClassification
#./explocal/exp1_systematic/runExperiment1.sh sylvine BinaryClassification
#./explocal/exp1_systematic/runExperiment1.sh albert BinaryClassification
#./explocal/exp1_systematic/runExperiment1.sh evita BinaryClassification

# # Multiclass classification datasets:
# ####################################
# ./explocal/exp1_systematic/runExperiment1.sh digits MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh newsgroups MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh dilbert MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh fabert MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh robert MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh volkert MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh dionis MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh jannis MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh wallis MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh helena MulticlassClassification

# # Regression datasets:
# ######################
# ./explocal/exp1_systematic/runExperiment1.sh cadata Regression
# ./explocal/exp1_systematic/runExperiment1.sh flora Regression
# ./explocal/exp1_systematic/runExperiment1.sh yolanda Regression