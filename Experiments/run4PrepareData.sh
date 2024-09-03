#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/data"
data_out_path="${root_path}/data"
config_path="${root_path}/setup/config"

cd ${data_path}
unzip Accidents.zip  
unzip Airline.zip 
unzip Financial.zip
unzip Hockey.zip 
unzip IMDB_IJS.zip
unzip IMDB.zip
unzip Lahman-2014.zip
unzip Walmart-2014.zip
unzip Walmart.zip
unzip Yelp.zip

cd ${config_path}
source venv/bin/activate

CMD="python DatasetPrepare.py --dataset-root-path ${data_path} \
        --multi-table True \
        --data-out-path ${data_out_path}"

 
$CMD --dataset-name Accidents --target-attribute klas_nesreca --task-type multiclass --target-table nesreca 

cd ${root_path}