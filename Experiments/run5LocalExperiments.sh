#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;
mkdir -p catdb-results;

exp_path="$(pwd)"

# Add headers to log files
echo "dataset,task_type,platform,time" >> "${exp_path}/results/Experiment1_Data_Profile.dat"
echo "dataset,time" >> "${exp_path}/results/Experiment1_CSVDataReader.dat"
echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,task_type,time" >> "${exp_path}/results/Experiment1_LLM_Pipe_Gen.dat"
echo "dataset,iteration,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,task_type,time" >> "${exp_path}/results/Experiment1_LLM_Pipe_Test.dat"

echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,task_type,time,result" >> "${exp_path}/results/Experiment2_CatDB_LLM_Pipe_Run.dat"
echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment2_AutoML_Corresponding.dat"

echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment3_AutoML_1H.dat"
echo "dataset,source,time" >> "${exp_path}/results/Experiment3_Hand-craft.dat"

cd ${exp_path}

CMD=./explocal/exp0_statistics/runExperiment0.sh
#CMD=./explocal/exp1_catalog/runExperiment1.sh
#CMD=./explocal/exp2_micro_benchmark/runExperiment2.sh 
#CMD=./explocal/exp3_end_to_end/runExperiment3.sh 


$CMD oml_dataset_1_rnc binary test
$CMD oml_dataset_2_rnc binary test
$CMD oml_dataset_3_rnc binary test
$CMD oml_dataset_4_rnc binary test
$CMD oml_dataset_5_rnc binary test
$CMD oml_dataset_6_rnc binary test
$CMD oml_dataset_7_rnc binary test
$CMD oml_dataset_8_rnc binary test
$CMD oml_dataset_9_rnc binary test
$CMD oml_dataset_10_rnc binary test
$CMD oml_dataset_11_rnc binary test
$CMD oml_dataset_12_rnc binary test
$CMD oml_dataset_13_rnc binary test
$CMD oml_dataset_14_rnc binary test
$CMD oml_dataset_15_rnc binary test
$CMD oml_dataset_16_rnc binary test
$CMD oml_dataset_17_rnc binary test
$CMD oml_dataset_18_rnc binary test
$CMD oml_dataset_19_rnc binary test
$CMD oml_dataset_20_rnc binary test
$CMD oml_dataset_21_rnc binary test
$CMD oml_dataset_22_rnc binary test
$CMD oml_dataset_23_rnc binary test
$CMD oml_dataset_24_rnc binary test
$CMD oml_dataset_25_rnc binary test
$CMD oml_dataset_26_rnc binary test
$CMD oml_dataset_27_rnc binary test
$CMD oml_dataset_28_rnc binary test
$CMD oml_dataset_29_rnc binary test
$CMD oml_dataset_30_rnc binary test
$CMD oml_dataset_31_rnc binary test
$CMD oml_dataset_32_rnc binary test
$CMD oml_dataset_33_rnc binary test
$CMD oml_dataset_34_rnc binary test
$CMD oml_dataset_35_rnc binary test
$CMD oml_dataset_36_rnc binary test
$CMD oml_dataset_37_rnc binary test
$CMD oml_dataset_38_rnc binary test
$CMD oml_dataset_39_rnc binary test
$CMD oml_dataset_40_rnc binary test
$CMD oml_dataset_41_rnc binary test
$CMD oml_dataset_42_rnc binary test
$CMD oml_dataset_43_rnc binary test
$CMD oml_dataset_44_rnc binary test
$CMD oml_dataset_45_rnc binary test
$CMD oml_dataset_46_rnc binary test
$CMD oml_dataset_47_rnc binary test
$CMD oml_dataset_48_rnc binary test
$CMD oml_dataset_49_rnc binary test
$CMD oml_dataset_50_rnc binary test
$CMD oml_dataset_51_rnc binary test
$CMD oml_dataset_52_rnc binary test
$CMD oml_dataset_53_rnc binary test
$CMD oml_dataset_54_rnc binary test
$CMD oml_dataset_55_rnc binary test
$CMD oml_dataset_56_rnc binary test
$CMD oml_dataset_57_rnc binary test
$CMD oml_dataset_58_rnc binary test
$CMD oml_dataset_59_rnc binary test
$CMD oml_dataset_60_rnc binary test
$CMD oml_dataset_61_rnc binary test
$CMD oml_dataset_62_rnc binary test
$CMD oml_dataset_63_rnc binary test
$CMD oml_dataset_64_rnc binary test
$CMD oml_dataset_65_rnc binary test
$CMD oml_dataset_66_rnc binary test
$CMD oml_dataset_67_rnc binary test
$CMD oml_dataset_68_rnc binary test
$CMD oml_dataset_69_rnc binary test
$CMD oml_dataset_70_rnc binary test
$CMD oml_dataset_71_rnc binary test
$CMD oml_dataset_72_rnc binary test
$CMD oml_dataset_73_rnc binary test
$CMD oml_dataset_74_rnc binary test
$CMD oml_dataset_75_rnc binary test
$CMD oml_dataset_76_rnc binary test
$CMD oml_dataset_77_rnc binary test
$CMD oml_dataset_78_rnc binary test
$CMD oml_dataset_79_rnc binary test
$CMD oml_dataset_80_rnc binary test
$CMD oml_dataset_81_rnc binary test
$CMD oml_dataset_82_rnc binary test
$CMD oml_dataset_83_rnc binary test
$CMD oml_dataset_84_rnc binary test
$CMD oml_dataset_85_rnc binary test
$CMD oml_dataset_86_rnc binary test
$CMD oml_dataset_87_rnc binary test
$CMD oml_dataset_88_rnc binary test
$CMD oml_dataset_89_rnc binary test
$CMD oml_dataset_90_rnc binary test
$CMD oml_dataset_91_rnc binary test
$CMD oml_dataset_92_rnc binary test
$CMD oml_dataset_93_rnc binary test
$CMD oml_dataset_94_rnc binary test
$CMD oml_dataset_95_rnc binary test
$CMD oml_dataset_96_rnc binary test
$CMD oml_dataset_97_rnc binary test
$CMD oml_dataset_98_rnc binary test
$CMD oml_dataset_99_rnc binary test
$CMD oml_dataset_100_rnc binary test
$CMD oml_dataset_101_rnc binary test
$CMD oml_dataset_102_rnc binary test
$CMD oml_dataset_103_rnc binary test
$CMD oml_dataset_104_rnc binary test
$CMD oml_dataset_105_rnc binary test
$CMD oml_dataset_106_rnc binary test
$CMD oml_dataset_107_rnc binary test
$CMD oml_dataset_108_rnc binary test
$CMD oml_dataset_109_rnc binary test
$CMD oml_dataset_110_rnc binary test
$CMD oml_dataset_111_rnc binary test
$CMD oml_dataset_112_rnc binary test
$CMD oml_dataset_113_rnc binary test
$CMD oml_dataset_114_rnc binary test
$CMD oml_dataset_115_rnc binary test
$CMD oml_dataset_116_rnc binary test
$CMD oml_dataset_117_rnc binary test
$CMD oml_dataset_118_rnc binary test
$CMD oml_dataset_119_rnc binary test
$CMD oml_dataset_120_rnc binary test
$CMD oml_dataset_121_rnc binary test
$CMD oml_dataset_122_rnc binary test
$CMD oml_dataset_123_rnc binary test
$CMD oml_dataset_124_rnc binary test
$CMD oml_dataset_125_rnc binary test
$CMD oml_dataset_126_rnc binary test
$CMD oml_dataset_127_rnc binary test
$CMD oml_dataset_128_rnc binary test
$CMD oml_dataset_129_rnc binary test
$CMD oml_dataset_130_rnc binary test
$CMD oml_dataset_131_rnc binary test
$CMD oml_dataset_132_rnc binary test
$CMD oml_dataset_133_rnc binary test
$CMD oml_dataset_134_rnc binary test
$CMD oml_dataset_135_rnc binary test
$CMD oml_dataset_136_rnc binary test