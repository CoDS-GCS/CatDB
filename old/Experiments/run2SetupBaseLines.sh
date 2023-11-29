#!/bin/bash

#cleanup
root_path="$(pwd)"
path="$(pwd)/setup"

rm -rf "$path/JavaBaseline"
mkdir -p "$path/JavaBaseline"

# clone Apache SystemDS repository

cd ..
#build 
mvn clean package -P distribution

# # clean-up last libs
rm -rf "../JavaBaseline/lib"

# move the jars outside to be accessible by the run scripts
mv target/lib/ "${path}/JavaBaseline/lib"
mv target/JavaBaseline.jar "${path}/JavaBaseline/"

# cd .. # move to parent path
# rm -rf systemds #clean-up

# # compile and setup java baselines (GIO, SystemDS, and some other implementations over the SystemDS)
# cd "JavaBaselines"
# rm -rf target
# mvn clean package -P distribution

# #cleanup
# rm -rf "$path/JavaBaselines"
# mkdir -p "$path/JavaBaselines"

# mv target/lib/ "$path/JavaBaselines/"
# mv target/JavaBaselines.jar "$path/JavaBaselines/"
 