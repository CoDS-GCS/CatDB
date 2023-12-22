#!/bin/bash

#cleanup
root_path="$(pwd)"
path="$(pwd)/setup"
rm -rf "$path/Baselines"
mkdir -p "$path/Baselines"

cd $path

output_dir="${path}/setup/automl_test"
rm -rf $output_dir
mkdir $output_dir

# Install AutoML Benchmark is a tool for benchmarking AutoML frameworks on tabular data. 
# URL: https://github.com/openml/automlbenchmark

rm -rf automlbenchmark # clean-up
git clone https://github.com/openml/automlbenchmark.git --branch stable --depth 1 #clone benchmark
cd automlbenchmark

# Create a virtual environments to install the dependencies in:
rm -rf venv
python -m venv venv
source venv/bin/activate

# Then install the dependencies:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Update configs
automl_config_path=${root_path}"/explocal/exp1_systematic/automl_config"
automl_setup_path=${path}"/automlbenchmark"

cp -r ${automl_config_path}"/constraints.yaml" ${automl_setup_path}"/resources/" # update constraints
cp -r ${automl_config_path}"/catdb.yaml" ${automl_setup_path}"/resources/benchmarks/" # update benchmark onfig

declare -a automl_frameworks=("AutoGluon" "AutoGluon_bestquality" "AutoGluon_hq" "AutoGluon_gq" "lightautoml" "flaml" "H2OAutoML" "mljarsupervised" "mljarsupervised_compete" "constantpredictor" "RandomForest" "TunedRandomForest" "TPOT" "GAMA" "autosklearn" "autosklearn2")

for framework in "${automl_frameworks[@]}"; do
    python runbenchmark.py ${framework} test 2m --outdir $output_dir
done

# cd ..

# build and setup Python baseline
# cp -r Baselines/src/* ${path}"/Baselines/"

# cd ${path}"/Baselines/"
# pip install -r requirements.txt # install requirements
#------------------------------------------------------