import sys
import openml
from sklearn.model_selection import train_test_split
from pathlib import Path


class DownloadDatasets(object):
    def __init__(self, out_path):
        self.out_path = out_path

    def download_dataset(self, datasetID, ds_name):
        dataset = openml.datasets.get_dataset(datasetID, download_qualities=False)
        data, y, categorical_indicator, attribute_names = dataset.get_data()

        (nrows, ncols) = data.shape

        number_classes = 'N/A'
        n_classes = data[dataset.default_target_attribute].nunique()
        number_classes = f'{n_classes}'

        # if task.task_type.lower() != 'regression':
        #     n_classes = data[dataset.default_target_attribute].nunique()
        #     number_classes = f'{n_classes}'
        #     if n_classes == 2:
        #         task_type = "binary"
        #     else:
        #         task_type = "multiclass"
        # else:
        #     task_type = "regression"

        if data is not None and len(data) > 0:
            data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
            Path(f"{self.out_path}/{ds_name}").mkdir(parents=True, exist_ok=True)

            data_train.to_csv(f'{self.out_path}/{ds_name}/{ds_name}_train.csv', index=False)
            data_test.to_csv(f'{self.out_path}/{ds_name}/{ds_name}_test.csv', index=False)
            description = dataset.description
            if description is None:
                description = ''
            f = open(f'{self.out_path}/{ds_name}/description.txt', 'w')
            f.write(description)

        # print (dataset.task_type)
        return dataset.default_target_attribute, nrows, ncols, dataset.original_data_url, number_classes


if __name__ == '__main__':
    data_out_path = sys.argv[1]
    setting_out_path = sys.argv[2]
    download_ds = DownloadDatasets(out_path=data_out_path)

    # datasetIDs = [(273,'IMDB.drama','binary'),
    #               (40517,'20_newsgroups.drift','binary'),
    #               (45575,'Epsilon','binary'),
    #               (45672,'prostate','binary'),
    #               (45668,'bates_classif_100','binary'),
    #               (45693,'simulated_electricity','binary'),
    #               (42732,'sf-police-incidents','binary'),
    #               (42757,'KDDCup09-Appetency','binary'),
    #               (45570,'Higgs','binary'),
    #               (41103,'STL-10','multiclass'),
    #               (45579,'Microsoft','multiclass'),
    #               (41983,'CIFAR-100','multiclass'),
    #               (41989,'GTSRB-HOG03','multiclass'),
    #               (1179,'BNG(solar-flare)','multiclass'),
    #               (149,'CovPokElec','muliclass'),
    #               (42088,'beer_reviews','multiclass'),
    #               (42132,'Traffic_violations','muliclass'),
    #               (4549,'Buzzinsocialmedia_Twitter','regression'),
    #               (45081,'Tallo','regression')]
    
    datasetIDs = [ (45693,'simulated_electricity','binary', 1),
                   (23513,'KDD98','binary', 2),                  
                   (45570,'Higgs','binary', 3),
                   (45072,'airlines','binary', 4),
                  (40514,'BNG_credit_g','binary', 5),
                  (45579,'Microsoft','multiclass', 6),
                  (45056,'cmc','multiclass', 7),
                  (37,'diabetes','multiclass', 8),
                  (43476,'3-million-Sudoku-puzzles-with-ratings','multiclass', 9),
                  (155,'pokerhand','multiclass', 10),
                  (4549,'Buzzinsocialmedia_Twitter','regression', 11),
                  (45045,'delays_zurich_transport','regression', 12),
                  (44065,'nyc-taxi-green-dec-2016','regression', 13),
                  (44057,'black_friday','regression', 14),
                  (42080,'federal_election','regression', 15),
                  ]
    

    dataset_list = 'row,orig_dataset_name,dataset_name,nrows,ncols,file_format,task_type,number_classes,original_url,target_feature,description\n'
    script_list =""
    for (dataset_id,orig_ds_name,task_type, dataset_index) in datasetIDs:        
        print(f" Downloading Dataset: dataset name={orig_ds_name}, dataset ID={dataset_id} \n")
        ds_name=f"dataset_{dataset_index}"
        target, nrows, ncols, original_data_url, number_classes = download_ds.download_dataset(datasetID=dataset_id, ds_name=ds_name)

        config_strs = [f"- name: {ds_name}",
                       "  dataset:",
                       f"    train: \'{{user}}/data/{ds_name}/{ds_name}_train.csv\'",
                       f"    test: \'{{user}}/data/{ds_name}/{ds_name}_test.csv\'",
                       f"    target: {target}",
                       f"    type: {task_type}",
                       "  folds: 10",
                       "\n"]
        config_str = "\n".join(config_strs)

        yaml_file_local = f'{data_out_path}/{ds_name}/{ds_name}.yaml'
        f_local = open(yaml_file_local, 'w')
        f_local.write("--- \n \n")
        f_local.write(config_str)
        f_local.close()

        yaml_file_benchmark = f'{setting_out_path}/{ds_name}.yaml'
        f = open(yaml_file_benchmark, 'w')
        f.write("--- \n \n")
        f.write(config_str)
        f.close()

        dataset_list += f'{dataset_index},{orig_ds_name},{ds_name},{nrows},{ncols},csv,{task_type},{number_classes},{original_data_url},{target}, "OpenML (DatasetID={dataset_id})"\n'
        script_list += f'./explocal/exp1_systematic/runExperiment1.sh {ds_name} {task_type} \n'

    f = open(f'{data_out_path}/dataset_list.csv', 'w')
    f.write(dataset_list)

    f_script = open(f'{data_out_path}/script_list.sh', 'w')
    f_script.write(script_list)
