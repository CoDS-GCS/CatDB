import sys
import openml
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset

class DownloadDatasets(object):
    def __init__(self, out_path):
         self.out_path = out_path
         self.datalist = openml.datasets.list_datasets(output_format="dataframe")
         self.datalist = self.datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

    def download_dataset(self, maxNumberOfInstances, minNumberOfInstances, maxNumberOfFeatures, minNumberOfFeatures,
                         maxNumberOfClasses, minNumberOfClasses, NumberOfDatasets):

        list = self.datalist[(self.datalist.NumberOfInstances >= minNumberOfInstances) & (self.datalist.NumberOfInstances <= maxNumberOfInstances) &
                             (self.datalist.NumberOfFeatures >= minNumberOfFeatures) & (self.datalist.NumberOfFeatures <= maxNumberOfFeatures) &
                             (self.datalist.NumberOfClasses >= minNumberOfClasses) & (self.datalist.NumberOfClasses <= maxNumberOfClasses)].sort_values(['NumberOfInstances'], ascending=[False])

        list.to_csv(f'/home/saeed/Documents/list-[{maxNumberOfInstances}-{maxNumberOfFeatures}-{maxNumberOfClasses}].csv', index=False)
        list = list.head(n=NumberOfDatasets)


        if len(list) > 0:
            names = list['name'].tolist()
            dids = list['did'].tolist()
            for i in range(0, len(dids)):
                dataset = openml.datasets.get_dataset(dids[i])
                data, y, categorical_indicator, attribute_names = dataset.get_data()
                name = f'{names[i]}-[{maxNumberOfInstances}-{maxNumberOfFeatures}-{maxNumberOfClasses}]'
                data.to_csv(f'{self.out_path}/{name}.csv', index=False)
                description = dataset.description
                if description is None:
                    description = ''
                f = open(f'{self.out_path}/{name}-[description].txt', 'w')
                f.write(description)
        else:
            print("There is no dataset with the filter!")
        print(f'before = {len(self.datalist)} -- after={len(list)}')


if __name__ == '__main__':
    data_out_path = '/home/saeed/Documents/Github/CatDB/data/' #sys.argv[1]
    download_ds = DownloadDatasets(out_path=data_out_path)
    maxNumberOfInstances = [1000, 10000, 100000, 1000000]
    minNumberOfInstances =[100, 1001, 10001, 100001]
    maxNumberOfFeatures = [10, 20, 30, 40]
    minNumberOfFeatures = [5, 11, 21, 31]
    maxNumberOfClasses = [3, 4, 5, 6]
    minNumberOfClasses = [2, 3, 4, 5]
    NumberOfDatasets = 5

    for i in range(0, len(maxNumberOfInstances)):
        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>  {i}')
        download_ds.download_dataset(maxNumberOfInstances[i], minNumberOfInstances[i], maxNumberOfFeatures[i],
                                     minNumberOfFeatures[i], maxNumberOfClasses[i], minNumberOfClasses[i], NumberOfDatasets)







# print(datalist)
# print(f"First 10 of {len(datalist)} datasets...")
# datalist.head(n=10)
#
# # The same can be done with lesser lines of code
# openml_df = openml.datasets.list_datasets(output_format="dataframe")
# openml_df.head(n=10)
#
# dataset = openml.datasets.get_dataset(45705)
# eeg, *_ = dataset.get_data()
# print(eeg)