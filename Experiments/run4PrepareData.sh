#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/data"
data_out_path="${root_path}/data"
config_path="${root_path}/setup/config"
catalog_path="${root_path}/catalog"

# Extract data profile info - Multi-table datasets
cd "${root_path}/catalog"
unzip Accidents/data_profile.zip -d Accidents/
unzip Airline/data_profile.zip -d Airline/
unzip Financial/data_profile.zip -d Financial/
unzip IMDB-IJS/data_profile.zip -d IMDB-IJS/
unzip Yelp/data_profile.zip -d Yelp/

unzip EU-IT/data_profile.zip -d EU-IT/
unzip Halloween/data_profile.zip -d Halloween/
unzip Mid-Feed/data_profile.zip -d Mid-Feed/
unzip Utility/data_profile.zip -d Utility/
unzip Violations/data_profile.zip -d Violations/
unzip WiFi/data_profile.zip -d WiFi/


# cd ${data_path}

# unzip Accidents.zip  
# unzip Airline.zip 
# unzip Financial.zip
# unzip IMDB-IJS.zip
# unzip Yelp.zip

# unzip EU-IT.zip
# unzip Halloween.zip
# unzip Mid-Feed.zip
# unzip Utility.zip
# unzip Violations.zip
# unzip WiFi.zip

cd ${config_path}
source venv/bin/activate

CMD="python DatasetPrepare.py --dataset-root-path ${data_path} \
        --multi-table True \
        --data-out-path ${data_out_path} \
        --catalog-root-path ${catalog_path}"

 
# $CMD --dataset-name Accidents --target-attribute klas_nesreca --task-type multiclass --target-table nesreca --mtos True
# $CMD --dataset-name Airline --target-attribute ArrDel15 --task-type multiclass --target-table On_Time_On_Time_Performance_2016_1 --mtos True 
# $CMD --dataset-name Financial --target-attribute status --task-type multiclass --target-table loan --mtos True
# $CMD --dataset-name IMDB-IJS --target-attribute gender --task-type binary --target-table actors --mtos True
# $CMD --dataset-name Yelp --target-attribute stars --task-type regression --target-table Reviews --mtos True

# $CMD --dataset-name EU-IT --target-attribute 'Position ' --task-type multiclass --multi-table False
# $CMD --dataset-name Halloween --target-attribute 'What.is.your.age.group.' --task-type multiclass --multi-table False
# $CMD --dataset-name Mid-Feed --target-attribute B6 --task-type multiclass --multi-table False
# $CMD --dataset-name Utility --target-attribute CSRI --task-type multiclass --multi-table False
# $CMD --dataset-name Violations --target-attribute 'Disposition Description' --task-type multiclass --multi-table False
# $CMD --dataset-name WiFi --target-attribute TechCenter --task-type multiclass --multi-table False


cd ${root_path}
