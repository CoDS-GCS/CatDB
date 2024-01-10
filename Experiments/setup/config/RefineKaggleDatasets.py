import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def getConfigStr(ds_name, target, task_type):
    config_strs = [f"- name: {ds_name}",
                       "  dataset:",
                       f"    train: \'{{user}}/data/{ds_name}/{ds_name}_train.csv\'",
                       f"    test: \'{{user}}/data/{ds_name}/{ds_name}_test.csv\'",
                       f"    target: {target}",
                       f"    type: {task_type}",
                       "  folds: 1",
                       "\n"]
    config_str = "\n".join(config_strs)
    return config_str

def NYCDataset(ds_name: str, ds_path:str, out_path:str, target: str):
    ds = f'{ds_path}/NYC/yellow_tripdata_2016-03.csv'
    df = pd.read_csv(ds)

    data_train, data_test = train_test_split(df, test_size=0.3, random_state=42)    
    data_train.to_csv(f'{out_path}/NYC/NYC_train.csv', index=False)
    data_test.to_csv(f'{out_path}/NYC/NYC_test.csv', index=False)
    
    return getConfigStr(ds_name=ds_name, target=target, task_type='regression')


def USCarsDataset(ds_name: str, ds_path:str, out_path:str, target: str):
    ds = f'{ds_path}/USCars/used_cars_data.csv'
    df = pd.read_csv(ds)

    data_train, data_test = train_test_split(df, test_size=0.3, random_state=42)    
    data_train.to_csv(f'{out_path}/USCars/USCars_train.csv', index=False)
    data_test.to_csv(f'{out_path}/USCars/USCars_test.csv', index=False)
    
    return getConfigStr(ds_name=ds_name, target=target, task_type='regression')

def CanadaPricePredictionDataset(ds_name: str, ds_path:str, out_path:str, target: str):
    ds = f'{ds_path}/CanadaPricePrediction/amz_ca_price_prediction_dataset.csv'
    df = pd.read_csv(ds)

    data_train, data_test = train_test_split(df, test_size=0.3, random_state=42)    
    data_train.to_csv(f'{out_path}/CanadaPricePrediction/CanadaPricePrediction_train.csv', index=False)
    data_test.to_csv(f'{out_path}/CanadaPricePrediction/CanadaPricePrediction_test.csv', index=False)
    
    return getConfigStr(ds_name=ds_name, target=target, task_type='regression')

if __name__ == '__main__':
    data_source_name = sys.argv[1]
    target_attributed = sys.argv[2]
    ds_path = sys.argv[3]
    task_type = sys.argv[4]
    out_path = sys.argv[5]

    config_str = None
    if data_source_name == "NYC":
        config_str = NYCDataset(ds_name=data_source_name, ds_path=ds_path, out_path=out_path, target=target_attributed)
    elif data_source_name == "USCars":
        config_str = USCarsDataset(ds_name=data_source_name, ds_path=ds_path, out_path=out_path, target=target_attributed) 
    elif data_source_name == "CanadaPricePrediction":
        config_str = CanadaPricePredictionDataset(ds_name=data_source_name, ds_path=ds_path, out_path=out_path, target=target_attributed)       
    
    yaml_file_local = f'{out_path}/{data_source_name}/{data_source_name}.yaml'
    f_local = open(yaml_file_local, 'w')
    f_local.write("--- \n \n")
    f_local.write(config_str)
    f_local.close()
