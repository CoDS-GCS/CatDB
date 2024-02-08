#!/bin/bash

datasource_name=$1
exp_path="$(pwd)"
constraint=0
exp_path="$(pwd)"
log_file_name="${exp_path}/results/Experiment2_AutoML_Corresponding.dat"
output_dir="${exp_path}/results/automl_results"
user_dir=${exp_path}
logging=console:warning,app:info


while IFS="," read -r dataset time
    do
        if [ $datasource_name == $dataset ]
            then
            constraint=$time
        fi
done < "${exp_path}/explocal/exp2_micro_benchmark/corresponding_times.csv"


cd "${exp_path}/setup/automlbenchmark/"
source venv/bin/activate

declare -a frameworks=("AutoGluon" "H2OAutoML" "mljarsupervised" "RandomForest" "TPOT" "lightautoml" "autosklearn2")

for framework in "${frameworks[@]}"; do 
    AMLB="python runbenchmark.py ${framework} ${datasource_name} ${constraint} --outdir=${output_dir} --userdir=${user_dir} --logging=${logging}"

    # sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
    # sleep 3

    start=$(date +%s%N)
    $AMLB
    end=$(date +%s%N)
    echo ${datasource_name}","${framework}","$((($end - $start) / 1000000))","${constraint} >>$log_file_name
done

cd ${exp_path}