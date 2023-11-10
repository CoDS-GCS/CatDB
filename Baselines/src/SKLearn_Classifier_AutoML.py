from util.InputArgs import InputArgs
from autoML.SKLearnAutoML import SKLearnAutoML
from util.Reader import Reader
import sys

if __name__ == '__main__':
    inputs = InputArgs(sys.argv)

    reader = Reader(filename=inputs.dataset)
    df = reader.reader_CSV()

    skl_automl = SKLearnAutoML(time_left=int(inputs.time_left), per_run_time_limit=int(inputs.per_run_time_limit))
    skl_automl.split_data(df, target=inputs.target_attribute, test_size= 0.2, random_size= 42)
    accuracy, max_models = skl_automl.runClassifier()

    with open(inputs.log_file_name, "a") as logfile:
        dataset_name = inputs.dataset.split("/")
        name = dataset_name[len(dataset_name)-1]
        name = name.split(".")[0]
        logfile.write(f'SKLearn_Classifier_AutoML,{name},{accuracy},{inputs.time_left},{inputs.per_run_time_limit},{max_models}\n')
