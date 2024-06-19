import pandas as pd


def reader_CSV(filename):
    df = pd.read_csv(filename, low_memory=False)
    return df


class Dataset(object):
    def __init__(self,
                 dataset_name: str,
                 train_path: str,
                 test_path: str,
                 task_type: str,
                 target_attribute: str):
        self.dataset_name = dataset_name
        self.train_path = train_path
        self.test_path = test_path
        self.task_type = task_type
        self.target_attribute = target_attribute