#!/bin/bash

root_path="$(pwd)"
data_path="/home/saeed/Downloads/tmp/dataset/kaggle/"
data_out_path="${root_path}/data"
config_path="${root_path}/setup/config/"
benchmark_path="${root_path}/setup/automlbenchmark/resources/benchmarks"

# declare -a datasets_info=("${data_path}/horsecolic/horsecolic,horsecolic,dataset_1,SurgicalLesion,binary" \
#                           "${data_path}/credit-g/credit-g,credit-g,dataset_2,class,binary" \
#                           "${data_path}/albert/albert,albert,dataset_3,class,binary" \
#                           "${data_path}/Sonar/Sonar,Sonar,dataset_4,Class,binary" \
#                           "${data_path}/abalone/abalone,abalone,dataset_5,Rings,binary" \
#                           "${data_path}/poker/poker,poker,dataset_6,CLASS,binary") 

declare -a datasets_info=("${data_path}/health-insurance/health-insurance,health-insurance,dataset_1,Response,binary" \
                          "${data_path}/pharyngitis/pharyngitis,pharyngitis,dataset_2,radt,binary") 

cd ${config_path}
source venv/bin/activate

for dataset_info in "${datasets_info[@]}"; do
        SCRIPT="python DatasetPrepare.py\
                --dataset-info ${dataset_info} \
                --data-out-path ${data_out_path} \
                --setting-out-path ${benchmark_path}"

        echo $SCRIPT
        $SCRIPT
        echo "======================================================="
done

cd ${root_path}