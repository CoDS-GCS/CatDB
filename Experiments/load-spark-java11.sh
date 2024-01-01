#!/bin/bash

root_path="$(pwd)"
java -Xms145g -Xmx145g

JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin":$PATH

export SPARK_HOME="${root_path}/dependencies/spark-3.5.0-bin-hadoop3"
export PATH="$SPARK_HOME/bin:$PATH"
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH
export PYSPARK_PYTHON="$CONDA_PREFIX/envs/kglids/bin/python3.8"
export PYSPARK_DRIVER_PYTHON="$CONDA_PREFIX/envs/kglids/bin/python3.8"
cd $root_path