import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

import sys
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *

class CleanData(object):
    def __init__(self, dataset_name, target_attribute, task_type, train_path, test_path, output_dir):
        self.dataset_name = dataset_name
        self.target_attribute = target_attribute
        self.task_type = task_type
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = output_dir

    def get_column_info(self, column, col_name):
        column = pd.to_numeric(column, errors='ignore')
        column = column.convert_dtypes()
        column = column.astype(str) if column.dtype == object else column

        fd = 1

        # column_object = column.infer_objects()
        # if column_object.name == 'category':
        #     mask = 1
        # else:
        #     mask = 0
        # if self.task_type != "regression" and col_name == self.target_attribute:
        #     mask = 1

        mask = 0
        datatype = "STRING"
        if is_bool_dtype(column):
            datatype = "BOOL"
        elif is_numeric_dtype(column):
            dt = column.dtypes
            if dt == "Int64":
                datatype = "INT64"
            elif dt == "Int32":
                datatype = "INT32"
            elif dt == "Int16":
                datatype = "INT16"
            elif dt == "Float64":
                datatype = "FP64"
            elif dt == "Float32":
                datatype = "FP32"
            else:
                datatype = "FP64"
        else:
            mask = 1

        return mask, fd, datatype

    def run(self):
        print(f"\n**** SAGA Data Prepare {self.dataset_name}****\n")
        mask = ['mask']
        fd = ['fd']
        schema = ['Schema']
        df_train = pd.read_csv(self.train_path, na_values=[' ', '?', '-'])
        df_test = pd.read_csv(self.test_path, na_values=[' ', '?', '-'])
        columns = df_train.columns
        for col in columns:
            if col != self.target_attribute:
                m, f, dt = self.get_column_info(df_train[col], col)
                mask.append(m)
                fd.append(f)
                schema.append(dt)

        m, f, dt = self.get_column_info(df_train[self.target_attribute], self.target_attribute)
        mask.append(m)
        fd.append(f)
        schema.append(dt)

        X_train = df_train.drop(self.target_attribute, axis=1)
        y_train = df_train[self.target_attribute]
        X_test = df_test.drop(self.target_attribute, axis=1)
        y_test = df_test[self.target_attribute]

        X_train[self.target_attribute] = y_train
        X_test[self.target_attribute] = y_test

        df_meta = pd.DataFrame(columns=[f"c_{i}" for i in range(1, len(columns) + 2)])
        df_meta.loc[len(df_meta)] = schema
        df_meta.loc[len(df_meta)] = mask
        df_meta.loc[len(df_meta)] = fd

        # Save data:
        X_train.to_csv(f"{self.output_dir}/{self.dataset_name}_orig_train.csv", index=False, header=True)
        X_test.to_csv(f"{self.output_dir}/{self.dataset_name}_orig_test.csv", index=False, header=True)
        df_meta.to_csv(f"{self.output_dir}/{self.dataset_name}_meta.csv", index=False, header=False)
