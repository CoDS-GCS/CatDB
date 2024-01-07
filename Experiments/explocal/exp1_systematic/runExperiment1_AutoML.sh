#!/bin/bash

dataset=$1
constraint=$2
log_file_name=$3

exp_path="$(pwd)"
output_dir="${exp_path}/results/automl_results"
user_dir=${exp_path}
logging=console:warning,app:info

cd "${exp_path}/setup/automlbenchmark/"
source venv/bin/activate

declare -a benchmarks=("catdb_openml")
#declare -a constraints=("1h" "2h" "3h" "4h" "5h" "6h" "7h" "8h" "9h" "10h"
# declare -a frameworks=("AutoGluon" "AutoGluon_bestquality" "AutoGluon_hq" "AutoGluon_gq" "H2OAutoML" "mljarsupervised" "mljarsupervised_compete" "constantpredictor" "RandomForest" "TunedRandomForest" "TPOT")

declare -a frameworks=("AutoGluon" "H2OAutoML" "mljarsupervised" "constantpredictor" "RandomForest" "TPOT")
for framework in "${frameworks[@]}"; do
    AMLB="python runbenchmark.py ${framework} ${dataset} ${constraint} --outdir=${output_dir} --userdir=${user_dir} --logging=${logging}"
    echo "AMLB_SCRIPT=${AMLB}"

    sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
    sleep 3

    start=$(date +%s%N)
    $AMLB
    end=$(date +%s%N)
    echo ${dataset}","${framework}","$((($end - $start) / 1000000))","${constraint} >>$log_file_name
done

cd ${exp_path}