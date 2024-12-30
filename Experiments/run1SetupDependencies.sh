#!/bin/bash

# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt-get update
# sudo apt-get install -y openjdk-11-jdk-headless

# sudo apt-get install -y clang++
# sudo apt-get install -y python3-dev
# sudo apt-get install python3.9-venv # required for automlbenchmark
# sudo apt-get install python3.10-venv # required for CAAFE 
# sudo apt install -y python3-pip
# sudo pip install -y virtualenv
# sudo apt-get install -y wget
# sudo apt-get install -y unzip
sudo apt install -y maven

# root_path="$(pwd)"

# # Setup Apache Spark 
# mkdir -p "${root_path}/dependencies"
# cd "${root_path}/dependencies"

# rm -rf spark-3.5.0-bin-hadoop3.tgz # clean-up
# rm -rf spark-3.5.0-bin-hadoop3 # clean-up
# wget https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
# tar -xvzf spark-3.5.0-bin-hadoop3.tgz

cd ${root_path}

