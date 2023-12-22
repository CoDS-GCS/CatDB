
#!/bin/bash

# framework: TODO
# benchmark: TODO
# constraint: TODO

exp_path="$(pwd)"
mode=local
fold=10
input_dir="${exp_path}/data"
output_dir="${exp_path}/results/automl_results"
user_dir="${exp_path}/explocal/exp1_systematic/automl_config/"
parallel=task
setup=force
keep_scores=true
exit_on_error='-e'
logging=console:warning,app:info

rm -rf ${output_dir}

cd "${exp_path}/setup/automlbenchmark/"
source venv/bin/activate
rm -rf results
mkdir results

user_dir=${exp_path}

#AutoGluon

AMLB="python runbenchmark.py AutoGluon catdb 2m --userdir=${user_dir}"

echo $AMLB            
$AMLB            
