#!/bin/bash

exp_path="$(pwd)"
output_dir="${exp_path}/results/automl_results"
user_dir=${exp_path}
logging=console:warning,app:info

cd "${exp_path}/setup/automlbenchmark/"
source venv/bin/activate

declare -a benchmarks=("catdb_openml")
declare -a constraints=("1h") #"2h" "3h" "4h" "5h" "6h" "7h" "8h" "9h" "10h"
declare -a frameworks=("AutoGluon" "AutoGluon_bestquality" "AutoGluon_hq" "AutoGluon_gq" "H2OAutoML" "mljarsupervised" "mljarsupervised_compete" "constantpredictor" "RandomForest" "TunedRandomForest" "TPOT")

for constraint in "${constraints[@]}"; do
    for framework in "${frameworks[@]}"; do
        for benchmark in "${benchmarks[@]}"; do
            AMLB="python runbenchmark.py ${framework} ${benchmark} ${constraint} --outdir=${output_dir} --userdir=${user_dir} --logging=${logging}"
            echo "CATDB_SCRIPT=${AMLB}"
            $AMLB
        done           
    done
done