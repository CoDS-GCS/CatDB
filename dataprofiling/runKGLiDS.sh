#!/bin/bash

root_data=$1
data_source_name=$2
output_path="${root_data}/${data_source_name}/data_profile"

rm -rf ${output_path}
mkdir -p output_path

python kglidsplus_main.py --data-source-name ${data_source_name} --data-source-path "${root_data}/${data_source_name}" --output-path ${output_path}