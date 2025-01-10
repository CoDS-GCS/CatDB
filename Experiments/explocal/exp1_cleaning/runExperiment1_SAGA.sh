#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
task_type=$2

# clean-up 
mkdir -p "${data_path}/SAGA"
saga_data_path="${data_path}/data_space/${dataset}"

metadata_path="${saga_data_path}/${dataset}_meta.csv"
test_data_path="${saga_data_path}/${dataset}_orig_test.csv"
train_data_path="${saga_data_path}/${dataset}_aug_train.csv"

cd "${exp_path}/setup/Baselines/SAGA"

CMD="java -Xmx130g -Xms130g -Xmn11g \
    -cp SystemDS.jar:lib/* \
    -Dlog4j.configuration=file:log4j-silent.properties \
    org.apache.sysds.api.DMLScript \
    -exec singlenode \
    -debug \
    -config SystemDS-config.xml"


if [[ $task_type != "regression" ]]
    then
        task=evalClassification
    else
        task=evalRegression
fi        
sep=","

start=$(date +%s%N)

$CMD -f topkTest1.dml -stats -nvargs sep=$sep dirtyData=$train_data_path  metaData=$metadata_path\
    primitives=primitives.csv parameters=param.csv sample=0.1 topk=3 expectedIncrease=10 max_iter=15 rv=50\
    enablePruning=TRUE testCV=TRUE cvk=3 split=0.7 seed=-1 func=$task output=${saga_data_path}/ 2>&1  

$CMD -f evaluatePip.dml -stats -nvargs sep=$sep trainData=$train_data_path testData=$test_data_path metaData=$metadata_path input=${saga_data_path} logical=FALSE func=$task out=${saga_data_path} dsName=$dataset 2>&1 | tee ${saga_data_path}/screenEval.txt 

end=$(date +%s%N)

log_file_name="${exp_path}/results/Experiment1_SAGA_Cleaning.dat"
echo ${dataset}","$((($end - $start) / 1000000)) >>$log_file_name

# check SAGA run sucessfully
if [ ! -f "${saga_data_path}/${dataset}_test.csv" ]; then
    cp -r ${test_data_path} "${saga_data_path}/${dataset}_test.csv"
    cp -r ${train_data_path} "${saga_data_path}/${dataset}_train.csv"
    echo ${dataset} >> ${exp_path}/results/Experiment1_SAGA_Fail.dat
else
    CMD="python -Wignore mainSAGARewriteConfig.py --metadata-path ${saga_data_path}/${dataset}.yaml --dataset-path ${data_path}/data_space"
    $CMD   
fi

