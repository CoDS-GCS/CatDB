#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/data"
data_out_path="${root_path}/data"
config_path="${root_path}/setup/config"

# # Extract data profile info - Multi-table datasets
# cd "${root_path}/catalog"
# unzip Accidents/data_profile.zip -d Accidents/
# unzip Airline/data_profile.zip -d Airline/
# unzip Financial/data_profile.zip -d Financial/
# unzip IMDB-IJS/data_profile.zip -d IMDB-IJS/
# unzip IMDB/data_profile.zip -d IMDB/
# unzip Lahman-2014/data_profile.zip -d Lahman-2014/
# unzip Walmart-2014/data_profile.zip -d Walmart-2014/
# unzip Walmart/data_profile.zip -d Walmart/
# unzip Yelp/data_profile.zip -d Yelp/


cd ${data_path}

# # Multi-table datasets
# unzip Accidents.zip  
# unzip Airline.zip 
# unzip Financial.zip
# unzip IMDB-IJS.zip
# unzip IMDB.zip
# unzip Lahman-2014.zip
# unzip Walmart-2014.zip
# unzip Walmart.zip
# unzip Yelp.zip

# Datasets with duplicate categorical values values


cd ${config_path}
source venv/bin/activate

CMD="python DatasetPrepare.py --dataset-root-path ${data_path} \
        --multi-table True \
        --data-out-path ${data_out_path}"

 
# $CMD --dataset-name Accidents --target-attribute klas_nesreca --task-type multiclass --target-table nesreca 
# $CMD --dataset-name Airline --target-attribute ArrDel15 --task-type multiclass --target-table On_Time_On_Time_Performance_2016_1 
# $CMD --dataset-name Financial --target-attribute status --task-type multiclass --target-table loan 
# $CMD --dataset-name IMDB-IJS --target-attribute gender --task-type binary --target-table actors 
# $CMD --dataset-name IMDB --target-attribute sex --task-type binary --target-table actors 
# $CMD --dataset-name Lahman-2014 --target-attribute salary --task-type regression --target-table salaries 
# #$CMD --dataset-name Walmart-2014 --target-attribute klas_nesreca --task-type regression --target-table nesreca 
# $CMD --dataset-name Walmart --target-attribute units --task-type regression --target-table train 
# $CMD --dataset-name Yelp --target-attribute stars --task-type regression --target-table Reviews 

#$CMD --dataset-name TSM-Habitat --target-attribute Stratum --task-type multiclass --multi-table False
#$CMD --dataset-name Violations --target-attribute 'Disposition Description' --task-type multiclass --multi-table False
#$CMD --dataset-name Mid-Feed --target-attribute B6 --task-type multiclass --multi-table False
#$CMD --dataset-name Utility --target-attribute CSRI --task-type multiclass --multi-table False
#$CMD --dataset-name EU-IT --target-attribute 'Position ' --task-type multiclass --multi-table False
#$CMD --dataset-name Health-Sciences --target-attribute 'Does your lab/research group currently use a naming convention to save your data files? ' --task-type multiclass --multi-table False
#$CMD --dataset-name Salaries --target-attribute 'job_title_category' --task-type multiclass --multi-table False
#$CMD --dataset-name Halloween --target-attribute 'What.is.your.age.group.' --task-type multiclass --multi-table False
#$CMD --dataset-name Relocated-Vehicles --target-attribute 'Relocated To Direction' --task-type multiclass --multi-table False
#$CMD --dataset-name Midwest-Survey --target-attribute 'Location (Census Region)' --task-type multiclass --multi-table False
#$CMD --dataset-name US-Labor --target-attribute case_status --task-type multiclass  --multi-table False
#$CMD --dataset-name Mental-Health --target-attribute 'How easy is it for you to take medical leave for a mental health condition?' --task-type multiclass --multi-table False
#$CMD --dataset-name San-Francisco --target-attribute TotalPay --task-type multiclass --multi-table False
# $CMD --dataset-name WiFi --target-attribute TechCenter --task-type multiclass --multi-table False
# $CMD --dataset-name Etailing --target-attribute 'What is the maximum cart value you ever shopped?' --task-type multiclass --multi-table False

cd ${root_path}
