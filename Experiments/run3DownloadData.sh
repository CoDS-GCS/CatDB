#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/data"
cd ${data_path}

# mkdir -p tmpdata
# cd tmpdata

#  # Download AutoML Datasets
#  wget http://www.causality.inf.ethz.ch/AutoML/adult.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/cadata.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/digits.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/dorothea.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/newsgroups.zip

#  wget http://www.causality.inf.ethz.ch/AutoML/christine.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/jasmine.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/philippine.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/madeline.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/sylvine.zip

#  wget http://www.causality.inf.ethz.ch/AutoML/albert.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/dilbert.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/fabert.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/robert.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/volkert.zip

#  wget http://www.causality.inf.ethz.ch/AutoML/alexis.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/dionis.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/grigoris.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/jannis.zip
#  wget http://www.causality.inf.ethz.ch/AutoML/wallis.zip

#  # Evita
#  wget https://competitions.codalab.org/my/datasets/download/c8fb35a9-8fa6-4627-90dc-e1210501c378
#  mv c8fb35a9-8fa6-4627-90dc-e1210501c378 evita.zip


#  # Flora
#  wget https://competitions.codalab.org/my/datasets/download/9b0e2bc2-7a5b-4513-8718-d234bc13bca2
#  mv 9b0e2bc2-7a5b-4513-8718-d234bc13bca2 flora.zip

#  # Helena
#  wget https://competitions.codalab.org/my/datasets/download/09ada795-4052-4fac-957a-87f02229b201
#  mv 09ada795-4052-4fac-957a-87f02229b201 helena.zip

#  # Tania
#  wget https://competitions.codalab.org/my/datasets/download/e52ae5cb-ba0b-4f56-92e4-974c63a855e3
#  mv e52ae5cb-ba0b-4f56-92e4-974c63a855e3 tania.zip

#  # Yolanda
#  wget https://competitions.codalab.org/my/datasets/download/41847153-1338-4514-a693-547f1288e8c4
#  mv 41847153-1338-4514-a693-547f1288e8c4 yolanda.zip


#  unzip adult.zip -d ../adult
#  unzip cadata.zip -d ../cadata
#  unzip digits.zip -d ../digits
#  unzip dorothea.zip -d ../dorothea
#  unzip newsgroups.zip -d ../newsgroups

#  unzip christine.zip -d ../christine
#  unzip jasmine.zip -d ../jasmine
#  unzip philippine.zip -d ../philippine
#  unzip madeline.zip -d ../madeline
#  unzip sylvine.zip -d ../sylvine

#  unzip albert.zip -d ../albert
#  unzip dilbert.zip -d ../dilbert
#  unzip fabert.zip -d ../fabert
#  unzip robert.zip -d ../robert
#  unzip volkert.zip -d ../volkert

#  unzip alexis.zip -d ../alexis
#  unzip dionis.zip -d ../dionis
#  unzip grigoris.zip -d ../grigoris
#  unzip jannis.zip -d ../jannis
#  unzip wallis.zip -d ../wallis

#  unzip evita.zip -d ../evita
#  unzip flora.zip -d ../flora
#  unzip helena.zip -d ../helena
#  unzip tania.zip -d ../tania
#  unzip yolanda.zip -d ../yolanda

cd ${data_path}
# rm -rf venv
# python -m venv venv
source venv/bin/activate

# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt


# Binary classification datasets: dorothea, christine, jasmine, philippine, madeline, sylvine, albert, evita
# Multiclass classification datasets: digits, newsgroups, dilbert, fabert, robert, volkert, dionis, jannis, wallis, helena
# Regression datasets: cadata, flora, yolanda
#"evita"

#declare -a datasets=("dorothea" "christine" "jasmine" "philippine" "madeline" "sylvine" "albert" "digits" "newsgroups" "dilbert" "fabert" "robert" "volkert" "dionis" "jannis" "wallis" "helena" "cadata" "flora" "yolanda")
#for dataset in "${datasets[@]}"; do
#    python RefineOpenMLDatasets.py ${data_path} ${dataset}
#    cp "${data_path}/${dataset}/${dataset}.yaml" "${root_path}/setup/automlbenchmark/resources/benchmarks/"
#done


# benchmark_path="${root_path}/setup/automlbenchmark/resources/benchmarks"
# python DownloadOpenMLDatasetsByTaskID.py ${data_path} ${benchmark_path}

benchmark_path="${root_path}/setup/automlbenchmark/resources/benchmarks"
python DownloadOpenMLDatasetsByDatasetID.py ${data_path}

cd ${root_path}


#Amazon ML Challenge Dataset 2023
#https://www.kaggle.com/datasets/kushagrathisside/amazon-ml-challenge-dataset-2023

#US Used cars dataset (9.98 GB)
#https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset

#Canada Optimal Product Price Prediction Dataset
#https://www.kaggle.com/datasets/asaniczka/canada-optimal-product-price-prediction