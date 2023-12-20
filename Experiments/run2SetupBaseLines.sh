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

rm -rf automlbenchmark # clean-up
git clone https://github.com/openml/automlbenchmark.git --branch stable --depth 1 #clone benchmark
cd automlbenchmark

# Create a virtual environments to install the dependencies in:
python -m venv venvAutoMLBenchmark
source venvAutoMLBenchmark/bin/activate

# Then install the dependencies:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# a test run with Random Forest
python runbenchmark.py randomforest 







# cd ..

# build and setup Python baseline
# cp -r Baselines/src/* ${path}"/Baselines/"

# cd ${path}"/Baselines/"
# pip install -r requirements.txt # install requirements
#------------------------------------------------------



