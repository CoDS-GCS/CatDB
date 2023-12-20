#!/bin/bash

#cleanup
root_path="$(pwd)"
path="$(pwd)/setup"
rm -rf "$path/Baselines"
mkdir -p "$path/Baselines"

echo $path
cd $path

# Install AutoML Benchmark is a tool for benchmarking AutoML frameworks on tabular data. 
# URL: https://github.com/openml/automlbenchmark

#rm -rf automlbenchmark # clean-up
#git clone https://github.com/openml/automlbenchmark.git --branch stable --depth 1 #clone benchmark
cd automlbenchmark

# Create a virtual environments to install the dependencies in:
python -m venv venvAutoMLBenchmark
source venvAutoMLBenchmark/bin/activate

# Then install the dependencies:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Update configs
automl_config_path=${root_path}"/explocal/exp1_systematic/automlbenchmark_config"
automl_setup_path=${path}"/automlbenchmark"

cp -r ${automl_config_path}"/config.yaml" ${automl_setup_path}"/examples/custom/" # update cutom config
cp -r ${automl_config_path}"/constraints.yaml" ${automl_setup_path}"/examples/custom/" # update constraints
cp -r ${automl_config_path}"/constraints.yaml" ${automl_setup_path}"/resources/" # update constraints
cp -r ${automl_config_path}"/frameworks.yaml" ${automl_setup_path}"/examples/custom/" # update frameworks
cp -r ${automl_config_path}"/frameworks.yaml" ${automl_setup_path}"/resources/" # update frameworks
cp -r ${automl_config_path}"/filedatasets.yaml" ${automl_setup_path}"/examples/custom/benchmarks/" # update filedatasets (e.g., dataset path)

# a test run with Random Forest
python runbenchmark.py randomforest 

# cd ..

# build and setup Python baseline
# cp -r Baselines/src/* ${path}"/Baselines/"

# cd ${path}"/Baselines/"
# pip install -r requirements.txt # install requirements
#------------------------------------------------------



