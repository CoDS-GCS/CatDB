import sys
import pandas as pd

if __name__ == '__main__':
    data_source = sys.argv[1]
    ds_name = sys.argv[2]

    train_path = f'{data_source}/{ds_name}/{ds_name}_train.csv'
    test_path = f'{data_source}/{ds_name}/{ds_name}_test.csv'
    
    ds_train = pd.read_csv(train_path)
    ds_test = pd.read_csv(test_path)    