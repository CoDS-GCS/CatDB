#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/data"
data_out_path="${root_path}/data"
saga_data_out_path="${root_path}/data/data_space"
config_path="${root_path}/setup/config"
catalog_path="${root_path}/catalog"

# declare -a dataset_list=("Midwest-Survey" "WiFi" "Utility" "EU-IT" "Etailing" "Accidents" "Airline" "Financial" "IMDB-IJS" "Yelp" "EtoE-data/part-1" "EtoE-data/part-2" "EtoE-data/part-3")


# rm -rf ${catalog_path}
# unzip "${catalog_path}.zip"

# rm -rf "${data_path}/part-1"
# rm -rf "${data_path}/part-2"
# rm -rf "${data_path}/part-3"

# for ds in "${dataset_list[@]}"; do
#         rm -rf "${data_path}/${ds}" # clean-up dataset
#         unzip "${data_path}/${ds}.zip" -d "${data_path}/"
# done

# mv ${data_path}/part-1/* "${data_path}/"
# mv ${data_path}/part-2/* "${data_path}/"
# mv ${data_path}/part-3/* "${data_path}/"

# rm -rf "${data_path}/part-1"
# rm -rf "${data_path}/part-2"
# rm -rf "${data_path}/part-3"


# cd ${config_path}
# source venv/bin/activate

# CMD="python DatasetPrepare.py --dataset-root-path ${data_path} \
#         --multi-table True \
#         --data-out-path ${data_out_path} \
#         --catalog-root-path ${catalog_path}"

 
# $CMD --dataset-name Accidents --target-attribute klas_nesreca --task-type multiclass --target-table nesreca --mtos True
# $CMD --dataset-name Airline --target-attribute ArrDel15 --task-type multiclass --target-table On_Time_On_Time_Performance_2016_1 --mtos True 
# $CMD --dataset-name Financial --target-attribute status --task-type multiclass --target-table loan --mtos True
# $CMD --dataset-name IMDB-IJS --target-attribute gender --task-type binary --target-table actors --mtos True
# $CMD --dataset-name Yelp --target-attribute stars --task-type multiclass --target-table Reviews --mtos True

# $CMD --dataset-name EU-IT --target-attribute 'Position ' --task-type multiclass --multi-table False
# $CMD --dataset-name Utility --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name WiFi --target-attribute TechCenter --task-type binary --multi-table False
# $CMD --dataset-name Midwest-Survey --target-attribute 'Location (Census Region)' --task-type multiclass --multi-table False
# $CMD --dataset-name Etailing --target-attribute 'What is the maximum cart value you ever shopped?' --task-type multiclass --multi-table False


# $CMD --dataset-name Volkert-out-0.01-np-0-nc-0-mv-0 --target-attribute class --task-type multiclass --multi-table False 
# $CMD --dataset-name Volkert-out-0.02-np-0-nc-0-mv-0 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0.03-np-0-nc-0-mv-0 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0.04-np-0-nc-0-mv-0 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0.05-np-0-nc-0-mv-0 --target-attribute class --task-type multiclass --multi-table False

# $CMD --dataset-name Volkert-out-0-np-1-nc-180-mv-0.1 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0-np-1-nc-180-mv-0.2 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0-np-1-nc-180-mv-0.3 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0-np-1-nc-180-mv-0.4 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0-np-1-nc-180-mv-0.5 --target-attribute class --task-type multiclass --multi-table False

# $CMD --dataset-name Volkert-out-0.05-np-1-nc-180-mv-0.1 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0.05-np-1-nc-180-mv-0.2 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0.05-np-1-nc-180-mv-0.3 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0.05-np-1-nc-180-mv-0.4 --target-attribute class --task-type multiclass --multi-table False
# $CMD --dataset-name Volkert-out-0.05-np-1-nc-180-mv-0.5 --target-attribute class --task-type multiclass --multi-table False

# $CMD --dataset-name Utility-out-0.01-np-0-nc-0-mv-0 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0.02-np-0-nc-0-mv-0 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0.03-np-0-nc-0-mv-0 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0.04-np-0-nc-0-mv-0 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0.05-np-0-nc-0-mv-0 --target-attribute CSRI --task-type regression --multi-table False

# $CMD --dataset-name Utility-out-0-np-1-nc-12-mv-0.1 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0-np-1-nc-12-mv-0.2 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0-np-1-nc-12-mv-0.3 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0-np-1-nc-12-mv-0.4 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0-np-1-nc-12-mv-0.5 --target-attribute CSRI --task-type regression --multi-table False

# $CMD --dataset-name Utility-out-0.05-np-1-nc-12-mv-0.1 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0.05-np-1-nc-12-mv-0.2 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0.05-np-1-nc-12-mv-0.3 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0.05-np-1-nc-12-mv-0.4 --target-attribute CSRI --task-type regression --multi-table False
# $CMD --dataset-name Utility-out-0.05-np-1-nc-12-mv-0.5 --target-attribute CSRI --task-type regression --multi-table False

# Prepare Datasets for SAGA
mkdir -p ${saga_data_out_path}

declare -a dataset_list=("oml_dataset_2_rnc" "oml_dataset_4_rnc" "oml_dataset_5_rnc" "oml_dataset_33_rnc" "oml_dataset_34_rnc" "oml_dataset_35_rnc" "oml_dataset_20_rnc" "oml_dataset_6_rnc" "oml_dataset_3_rnc" "oml_dataset_22_rnc" "oml_dataset_24_rnc" "oml_dataset_23_rnc" "Airline" "IMDB-IJS" "Accidents" "Financial" "EU-IT" "Etailing" "Midwest-Survey" "WiFi" "Utility" "Yelp")

saga_baseline="${root_path}/setup/Baselines/SAGA"
cd ${saga_baseline}
source venv/bin/activate

CMD="python -Wignore mainSAGA.py"

for ds in "${dataset_list[@]}"; do
    output_dir="${saga_data_out_path}/${ds}/"
    metadata_path="${data_path}/${ds}/${ds}.yaml"
    #rm -rf ${output_dir} # clean-up dataset
    mkdir -p ${output_dir}

    cp -r ${metadata_path} "${output_dir}/${ds}_SAGA.yaml"    

    $CMD --metadata-path ${metadata_path} \
    --dataset-path ${data_path} \
    --output-dir ${output_dir}

done

cd ${root_path}
