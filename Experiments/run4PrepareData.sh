#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/data"
data_out_path="${root_path}/data"
config_path="${root_path}/setup/config"

# Extract data profile info
cd "${root_path}/catalog"
unzip Accidents/data_profile.zip -d Accidents/
unzip Airline/data_profile.zip -d Airline/
unzip Financial/data_profile.zip -d Financial/
# unzip Hockey/data_profile.zip -d Hockey/
unzip IMDB-IJS/data_profile.zip -d IMDB-IJS/
unzip IMDB/data_profile.zip -d IMDB/
unzip Lahman-2014/data_profile.zip -d Lahman-2014/
unzip Walmart-2014/data_profile.zip -d Walmart-2014/
unzip Walmart/data_profile.zip -d Walmart/
unzip Yelp/data_profile.zip -d Yelp/

cd ${data_path}
unzip Accidents.zip  
unzip Airline.zip 
unzip Financial.zip
# unzip Hockey.zip 
unzip IMDB-IJS.zip
unzip IMDB.zip
unzip Lahman-2014.zip
unzip Walmart-2014.zip
unzip Walmart.zip
unzip Yelp.zip

# cd ${config_path}
# source venv/bin/activate

# CMD="python DatasetPrepare.py --dataset-root-path ${data_path} \
#         --multi-table True \
#         --data-out-path ${data_out_path}"

 
# $CMD --dataset-name Accidents --target-attribute klas_nesreca --task-type multiclass --target-table nesreca 
# $CMD --dataset-name Airline --target-attribute ArrDel15 --task-type multiclass --target-table On_Time_On_Time_Performance_2016_1 
# $CMD --dataset-name Financial --target-attribute status --task-type multiclass --target-table loan 
# $CMD --dataset-name IMDB-IJS --target-attribute gender --task-type binary --target-table actors 
# $CMD --dataset-name IMDB --target-attribute sex --task-type binary --target-table actors 
# $CMD --dataset-name Lahman-2014 --target-attribute salary --task-type regression --target-table salaries 
# #$CMD --dataset-name Walmart-2014 --target-attribute klas_nesreca --task-type regression --target-table nesreca 
# $CMD --dataset-name Walmart --target-attribute units --task-type regression --target-table train 
# $CMD --dataset-name Yelp --target-attribute stars --task-type regression --target-table Reviews 

cd ${root_path}
