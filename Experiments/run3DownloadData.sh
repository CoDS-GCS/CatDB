#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/data"
config_path="${root_path}/setup/config/"

mkdir -p ${data_path}
cd ${config_path}
source venv/bin/activate

python DownloadOpenMLDatasetsByDatasetID.py --data-out-path ${data_path}
# python DownloadOpenMLDatasetsByTaskID.py --data-out-path ${data_path}

# python GenerateDataMissingValues.py --data-in-path "${config_path}/datasets" --data-out-path ${data_path}
# cd ${root_path}