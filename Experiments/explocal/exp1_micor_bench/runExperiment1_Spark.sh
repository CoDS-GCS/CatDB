#!/bin/bash

main_class=$1
dataset=$2
base_line=$3
log_file_name=$4


# Run Spark CSV
echo 3 >/proc/sys/vm/drop_caches

nrow=$(sed -n '$=' "data/${dataset}/yelp_academic_dataset_business.json")
echo $nrow

start=$(date +%s%N)
spark-submit  --executor-memory 28g  --driver-memory 20g --class ${main_class} --master local[*] "setup/JavaBaseline/JavaBaseline.jar" "data/${dataset}"
end=$(date +%s%N)

echo ${base_line}","${dataset}","$((($end - $start) / 1000000)) >>results/$log_file_name