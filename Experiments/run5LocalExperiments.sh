#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;
mkdir -p catdb-results;

exp_path="$(pwd)"

# Add headers to log files
echo "dataset,platform,time" >> "${exp_path}/results/Experiment1_Data_Profile.dat"
echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment1_AutoML.dat"
echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,prompt_number_iteration,task_type,time" >> "${exp_path}/results/Experiment1_CatDB.dat"
echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,prompt_number_iteration,task_type,time,result" >> "${exp_path}/results/Experiment1_CatDB_LLM_Code.dat"

./explocal/exp1_systematic/runExperiment1.sh airlines binary 0
./explocal/exp1_systematic/runExperiment1.sh albert binary 0
./explocal/exp1_systematic/runExperiment1.sh covertype multiclass 0
./explocal/exp1_systematic/runExperiment1.sh dionis multiclass 0
./explocal/exp1_systematic/runExperiment1.sh adult binary 0
./explocal/exp1_systematic/runExperiment1.sh Amazon_employee_access binary 0
./explocal/exp1_systematic/runExperiment1.sh APSFailure binary 0
./explocal/exp1_systematic/runExperiment1.sh bank-marketing binary 0
./explocal/exp1_systematic/runExperiment1.sh connect-4 multiclass 0
./explocal/exp1_systematic/runExperiment1.sh Fashion-MNIST multiclass 0
./explocal/exp1_systematic/runExperiment1.sh guillermo binary 0 # TODO: gpt-4 code gen error (limit request)
./explocal/exp1_systematic/runExperiment1.sh helena multiclass 0
./explocal/exp1_systematic/runExperiment1.sh higgs binary 0
./explocal/exp1_systematic/runExperiment1.sh jannis multiclass 0
./explocal/exp1_systematic/runExperiment1.sh jungle_chess_2pcs_raw_endgame_complete multiclass 0
./explocal/exp1_systematic/runExperiment1.sh KDDCup09_appetency binary 0
./explocal/exp1_systematic/runExperiment1.sh MiniBooNE binary 0
./explocal/exp1_systematic/runExperiment1.sh nomao binary 0
./explocal/exp1_systematic/runExperiment1.sh numerai28.6 binary 0
./explocal/exp1_systematic/runExperiment1.sh riccardo binary 0 # TODO: gpt-4 code gen error (limit request)
./explocal/exp1_systematic/runExperiment1.sh robert multiclass 0 # TODO: gpt-4 code gen error (limit request)
./explocal/exp1_systematic/runExperiment1.sh shuttle multiclass 0
./explocal/exp1_systematic/runExperiment1.sh volkert multiclass 0
./explocal/exp1_systematic/runExperiment1.sh Australian binary 0
./explocal/exp1_systematic/runExperiment1.sh blood-transfusion-service-center binary 0 
./explocal/exp1_systematic/runExperiment1.sh car multiclass 0 # TODO: fix pipeline error, gpt-4
./explocal/exp1_systematic/runExperiment1.sh christine binary 0
./explocal/exp1_systematic/runExperiment1.sh cnae-9 multiclass 0
./explocal/exp1_systematic/runExperiment1.sh credit-g binary 0
./explocal/exp1_systematic/runExperiment1.sh dilbert multiclass 0
./explocal/exp1_systematic/runExperiment1.sh fabert multiclass 0
./explocal/exp1_systematic/runExperiment1.sh jasmine binary 0
./explocal/exp1_systematic/runExperiment1.sh kc1 binary 0
./explocal/exp1_systematic/runExperiment1.sh kr-vs-kp binary 0 # TODO: fix dimention error in gpt-3.5 
./explocal/exp1_systematic/runExperiment1.sh mfeat-factors multiclass 0 
./explocal/exp1_systematic/runExperiment1.sh phoneme binary 0
./explocal/exp1_systematic/runExperiment1.sh segment multiclass 0 
./explocal/exp1_systematic/runExperiment1.sh sylvine binary 0
./explocal/exp1_systematic/runExperiment1.sh vehicle multiclass 0 

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