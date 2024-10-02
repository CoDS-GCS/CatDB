import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetReader(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = None

    def reader_CSV(self):
        df = pd.read_csv(self.filename, low_memory=False)
        return df

    def reader_JSON(self):
        df = pd.read_json(self.filename, lines=True)
        return df


def split_clean_data_save(data_path, ds_name, out_path):
    from util.Config import _llm_platform
    data =  pd.read_csv(data_path, low_memory=False)
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
    _, data_verify = train_test_split(data_train, test_size=0.1, random_state=42)

    data_train.to_csv(f'{out_path}/{ds_name}/{ds_name}_train_{_llm_platform}_clean.csv', index=False)
    data_test.to_csv(f'{out_path}/{ds_name}/{ds_name}_test_{_llm_platform}_clean.csv', index=False)
    data_verify.to_csv(f'{out_path}/{ds_name}/{ds_name}_verify_{_llm_platform}_clean.csv', index=False)


def split_manual_clean_data_save(data, ds_name, out_path):
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
    _, data_verify = train_test_split(data_train, test_size=0.1, random_state=42)

    data_train.to_csv(f'{out_path}/{ds_name}/{ds_name}_train_manual_clean.csv', index=False)
    data_test.to_csv(f'{out_path}/{ds_name}/{ds_name}_test_manual_clean.csv', index=False)
    data_verify.to_csv(f'{out_path}/{ds_name}/{ds_name}_verify_manual_clean.csv', index=False)