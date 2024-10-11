#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/data"
data_out_path="${root_path}/data"
config_path="${root_path}/setup/config"
catalog_path="${root_path}/catalog"

declare -a dataset_list=("Midwest-Survey" "WiFi" "Utility" "EU-IT" "Etailing" "Accidents" "Airline" "Financial" "IMDB-IJS" "Yelp")

rm -rf ${catalog_path}
unzip "${catalog_path}.zip"

for ds in "${dataset_list[@]}"; do
        rm -rf "${data_path}/${ds}" # clean-up dataset
        unzip "${data_path}/${ds}.zip" -d "${data_path}/"
done

cd ${config_path}
source venv/bin/activate

CMD="python DatasetPrepare.py --dataset-root-path ${data_path} \
        --multi-table True \
        --data-out-path ${data_out_path} \
        --catalog-root-path ${catalog_path}"

 
$CMD --dataset-name Accidents --target-attribute klas_nesreca --task-type multiclass --target-table nesreca --mtos True
$CMD --dataset-name Airline --target-attribute ArrDel15 --task-type multiclass --target-table On_Time_On_Time_Performance_2016_1 --mtos True 
$CMD --dataset-name Financial --target-attribute status --task-type multiclass --target-table loan --mtos True
$CMD --dataset-name IMDB-IJS --target-attribute gender --task-type binary --target-table actors --mtos True
$CMD --dataset-name Yelp --target-attribute stars --task-type multiclass --target-table Reviews --mtos True

$CMD --dataset-name EU-IT --target-attribute 'Position ' --task-type multiclass --multi-table False
$CMD --dataset-name Utility --target-attribute CSRI --task-type regression --multi-table False
$CMD --dataset-name WiFi --target-attribute TechCenter --task-type binary --multi-table False
$CMD --dataset-name Midwest-Survey --target-attribute 'Location (Census Region)' --task-type multiclass --multi-table False
$CMD --dataset-name Etailing --target-attribute 'What is the maximum cart value you ever shopped?' --task-type multiclass --multi-table False

# $CMD --dataset-name Relocated-Vehicles --target-attribute 'Relocated To Direction' --task-type multiclass --multi-table False
# $CMD --dataset-name Health-Sciences --target-attribute 'Does your lab/research group currently use a naming convention to save your data files? ' --task-type multiclass --multi-table False
# $CMD --dataset-name Violations --target-attribute 'Disposition Description' --task-type multiclass --multi-table False
# $CMD --dataset-name Halloween --target-attribute 'What.is.your.age.group.' --task-type multiclass --multi-table False
# $CMD --dataset-name Mid-Feed --target-attribute B6 --task-type multiclass --multi-table False

cd ${root_path}
