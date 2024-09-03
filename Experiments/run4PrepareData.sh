#!/bin/bash

root_path="$(pwd)"
data_path="/home/saeed/Downloads/RDB-Export/"
data_out_path="${root_path}/data"
config_path="${root_path}/setup/config"

cd ${config_path}
source venv/bin/activate

CMD="python DatasetPrepare.py --dataset-root-path ${data_path} \
        --multi-table True \
        --data-out-path ${data_out_path}"

$CMD --dataset-name Accidents --target-attribute klas_nesreca --task-type multiclass --target-table nesreca 

cd ${root_path}