import sys
import openml
import pandas as pd

class DownloadDatasets(object):
    def __init__(self, out_path):
         self.out_path = out_path
         self.datalist = openml.datasets.list_datasets(output_format="dataframe")
         self.datalist = self.datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

    def download_dataset(self, NumberOfDatasets):

        list = self.datalist[(self.datalist.NumberOfClasses == 2) & (self.datalist.NumberOfFeatures <= 200)].sort_values(['NumberOfInstances'], ascending=[False])
        list = list.head(n=NumberOfDatasets)

        ds_list =  pd.DataFrame(columns=["ID","dataset_name","nrows","ncols","nclasses","target"])

        if len(list) > 0:
            names = list['name'].tolist()
            dids = list['did'].tolist()
            for i in range(0, len(dids)):
                dataset = openml.datasets.get_dataset(dids[i])
                data, y, categorical_indicator, attribute_names = dataset.get_data()
                target_attribute = dataset.default_target_attribute
                name = f'{names[i]}'
                (nrows, ncols) = data.shape
                data.to_csv(f'{self.out_path}/{name}.csv', index=False)
                ds_list.loc[len(ds_list)] = [dids[i],name,nrows,ncols,2,target_attribute]
                ds_list.to_csv(f"{self.out_path}/dataset_list.csv")


if __name__ == '__main__':
    data_out_path = sys.argv[1]

    download_ds = DownloadDatasets(out_path=data_out_path)
    NumberOfDatasets = 200

    download_ds.download_dataset(NumberOfDatasets=NumberOfDatasets)