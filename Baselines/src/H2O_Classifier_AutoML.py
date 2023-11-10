from util.InputArgs import InputArgs
from autoML.H2O_AutoML import H2O_AutoML
import sys

if __name__ == '__main__':
    inputs = InputArgs(sys.argv)

    h2o_automl = H2O_AutoML( time_left=int(inputs.time_left), per_run_time_limit=int(inputs.per_run_time_limit))
    h2o_automl.split_data(inputs.dataset, target=inputs.target_attribute, test_size= 0.2, random_size= .15)
    accuracy, max_models = h2o_automl.runClassifier()

    with open(inputs.log_file_name, "a") as logfile:
        dataset_name = inputs.dataset.split("/")
        name = dataset_name[len(dataset_name)-1]
        name = name.split(".")[0]
        logfile.write(f'H2O_Classifier_AutoML,{name},{accuracy},{inputs.time_left},{inputs.per_run_time_limit},{max_models}\n')
