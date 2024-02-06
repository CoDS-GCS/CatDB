#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/cleaning-datasets"
data_out_path="${root_path}/data"
config_path="${root_path}/setup/config/"
benchmark_path="${root_path}/setup/automlbenchmark/resources/benchmarks"

declare -a datasets_info=("${data_path}/horsecolic/horsecolic,horsecolic,dataset_1,SurgicalLesion,binary" \
                          "${data_path}/credit-g/credit-g,credit-g,dataset_2,class,binary" \
                          "${data_path}/albert/albert,albert,dataset_3,class,binary" \
                          "${data_path}/Sonar/Sonar,Sonar,dataset_4,Class,binary" \
                          "${data_path}/abalone/abalone,abalone,dataset_5,Rings,binary" \
                          "${data_path}/poker/poker,poker,dataset_6,CLASS,binary") 

cd ${config_path}
source venv/bin/activate

for dataset_info in "${datasets_info[@]}"; do
        SCRIPT="python DatasetPrepare.py\
                --dataset-info ${dataset_info} \
                --data-out-path ${data_out_path} \
                --setting-out-path ${benchmark_path}"

        echo $SCRIPT
        $SCRIPT
        echo "======================================================="
done

cd ${root_path}


# for cleaning: /mnt/Niki/datasets_og.csv 
# for transformation: 'robert':'class', 'dilbert':'class','guillermo':'class','labor':'17','hepatitis_trans':'Class','Ionosphere':'column_ai','Sonar':'Class','APomentumlung':'Tissue','APomentumovary':'Tissue','Gisette':'Class','Spectfheart':'OVERALL_DIAGNOSIS','Madelon':'Class','Spambase':'Class','Credita':'A16','Pimaindianssubset':'Outcome','diabetes':'Outcome','covertype':'class','dionis':'class','Fashion-MNIST':'class','bng_pbc':'target','OVA_Breast':'Tissue','mnist_784':'class',"bank-marketing":"Class","christine":"class","MiniBooNE":"signal","nomao":"Class","volkert":"class",'abalone':'Rings','arcene_train':'class','banknote_authentication':'class','dermatology':'class','ecoli':'class','fertility_Diagnosis':'diagnosis','haberman':'survival','letter_recognition':'letter','libras':'class','poker':'CLASS','shuttle':'class','waveform':'class','wine':'class','featurefourier':'class','featurepixel':'class','opticaldigits':'class' and binary classification

