#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/data"
cd ${data_path}

mkdir -p tmpdata
cd tmpdata
unzip NYC.zip -d NYC
unzip USCars.zip -d USCars
unzip CanadaPricePrediction.zip -d CanadaPricePrediction
cd ..

cd ${data_path}
# rm -rf venv
# python -m venv venv
source venv/bin/activate

# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt

# benchmark_path="${root_path}/setup/automlbenchmark/resources/benchmarks"
# python DownloadOpenMLDatasetsByTaskID.py ${data_path} ${benchmark_path}

# benchmark_path="${root_path}/setup/automlbenchmark/resources/benchmarks"
# python DownloadOpenMLDatasetsByDatasetID.py ${data_path}

# Refine kaggle datasets
mkdir -p "${data_path}/NYC"
mkdir -p "${data_path}/USCars"
mkdir -p "${data_path}/CanadaPricePrediction"

python RefineKaggleDatasets.py NYC Total_amount "${data_path}/tmpdata/" regression ${data_path}
python RefineKaggleDatasets.py USCars price "${data_path}/tmpdata/" regression ${data_path}
python RefineKaggleDatasets.py CanadaPricePrediction price "${data_path}/tmpdata/" regression ${data_path}

cd ${root_path}

#NYC Yellow Taxi Trip Data
#https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data

#Amazon ML Challenge Dataset 2023
#https://www.kaggle.com/datasets/kushagrathisside/amazon-ml-challenge-dataset-2023

#US Used cars dataset (9.98 GB)
#https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset

#Canada Optimal Product Price Prediction Dataset
#https://www.kaggle.com/datasets/asaniczka/canada-optimal-product-price-prediction