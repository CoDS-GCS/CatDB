#!/bin/bash

#cleanup
root_path="$(pwd)"
path="$(pwd)/setup"
mkdir -p "$path/Baselines"

cd $path

## Install AutoML Benchmark is a tool for benchmarking AutoML frameworks on tabular data. 
## URL: https://github.com/openml/automlbenchmark

# rm -rf automlbenchmark # clean-up
# git clone https://github.com/openml/automlbenchmark.git --branch stable --depth 1 #clone benchmark
# cd automlbenchmark

# ## Create a virtual environments to install the dependencies in:
# rm -rf venv
# python -m venv venv
# source venv/bin/activate

# ## Then install the dependencies:
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt

# cp -r ${path}"/config/automl/constraints.yaml" ${path}"/automlbenchmark/resources/" # update constraints
# cp -r ${path}"/config/automl/frameworks.yaml" ${path}"/automlbenchmark/resources/" # update frameworks
# cp -r ${path}"/config/automl/config.yaml" ${path}"/automlbenchmark/resources/" # update config

# python runbenchmark.py AutoGluon --setup=only
# python runbenchmark.py AutoGluon_bestquality --setup=only
# python runbenchmark.py AutoGluon_hq --setup=only
# python runbenchmark.py AutoGluon_gq --setup=only
# python runbenchmark.py H2OAutoML --setup=only
# python runbenchmark.py mljarsupervised --setup=only
# python runbenchmark.py mljarsupervised_compete --setup=only
# python runbenchmark.py constantpredictor --setup=only
# python runbenchmark.py RandomForest --setup=only
# python runbenchmark.py TunedRandomForest --setup=only
# python runbenchmark.py TPOT --setup=only

# ## TODO: fix the following frameworks error
# python runbenchmark.py lightautoml --setup=only
# python runbenchmark.py flaml --setup=only
# python runbenchmark.py GAMA --setup=only
# python runbenchmark.py autosklearn --setup=only
# python runbenchmark.py autosklearn2 --setup=only


# Setup kglids
##############

# cd "${path}/Baselines"
# rm -rf kglids
# git clone https://github.com/CoDS-GCS/kglids.git

# cd kglids
# conda create -n kglids python=3.8 -y
# eval "$(conda shell.bash hook)"
# conda activate kglids
# python -m pip install --upgrade pip
# pip install -r requirements.txt

# cp "${root_path=}/data/kglids_main.py"  "${path}/Baselines/kglids/kg_governor/data_profiling/src/"


# Setup CatDB
#############
# catdb_path="${path}/Baselines/CatDB/"
# rm -rf ${catdb_path}
# mkdir -p ${catdb_path}

# cd ${root_path}
# cd ..
# cp -r src/python/main/* ${catdb_path}
# cd ${catdb_path}

# rm -rf venv
# python -m venv venv
# source venv/bin/activate

# # Then install the dependencies:
# python -m pip install --upgrade pip
# pip install torchvision 
# python -m pip install -r requirements.txt
# pip install pipreqs

# Prepare Config
################
# config_path="${path}/config/"
# cd ${config_path}
# rm -rf venv
# python -m venv venv
# source venv/bin/activate

# #Then install the dependencies:
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt


# Setup Hand-craft baseline
################
# baselines_path="${root_path}/baselines/"
# cd ${baselines_path}
# rm -rf venv
# python -m venv venv
# source venv/bin/activate

# # Then install the dependencies:
# pip install pipreqs
# pipreqs --force --mode no-pin ${baselines_path}
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt


# # Setup CAFFE
# baselines_path="$path/Baselines"
# cd ${baselines_path}
# rm -rf CAAFE
# git clone https://github.com/automl/CAAFE.git

# cd CAAFE
# rm -rf venv
# python3.9 -m venv venv
# source venv/bin/activate
# python3.9 -m pip install --upgrade pip
# cp "${path}/config/CAAFE/setup.py" "${baselines_path}/CAAFE/"
# python3.9 setup.py install

# ==============================================================================
# Setup CAFFE
baselines_path="$path/Baselines"
cd ${baselines_path}
rm -rf CAAFE
cp -r ${path}"/config/CAAFE" ${baselines_path}
cd CAAFE

rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# echo "-----------------------------------------------------------------------"
# pip --version
# pip install lightgbm


