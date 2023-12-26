#!/bin/bash
root_path="$(pwd)"
data_path="${root_path}/data"

# Enable venv for download datasets and resources
cd ${data_path}
rm -rf venv
python -m venv venv
source venv/bin/activate

# Download fasttext pre-trained models and save it in src/main/python/resources path
cd $root_path
cd ..

cd src/main/python/resources
python DownloadFasttext.py # Run python script for download models

cd ${data_path} # Got to data directory and download experiment datasets

ls -hl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python DownloadOpenMLDatasetsByTaskID.py ${data_path}

mv "${data_path}/catdb_openml.yaml" "${root_path}/setup/automlbenchmark/resources/benchmarks/"

cd ${root_path}