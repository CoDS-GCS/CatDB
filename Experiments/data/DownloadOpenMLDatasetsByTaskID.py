import sys
import openml
from sklearn.model_selection import train_test_split
from pathlib import Path


class DownloadDatasets(object):
    def __init__(self, out_path):
        self.out_path = out_path

    def download_dataset(self, taskID):
        task = openml.tasks.get_task(taskID, download_qualities=False)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_qualities=False)
        data, y, categorical_indicator, attribute_names = dataset.get_data()
        ds_name = dataset.name

        (nrows, ncols) = data.shape

        number_classes = 'N/A'
        if task.task_type.lower() != 'regression':
            number_classes = f'{data[dataset.default_target_attribute].nunique()}'

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



        return ds_name, dataset.default_target_attribute, nrows, ncols, dataset.dataset_id, dataset.original_data_url, task.task_type, number_classes


if __name__ == '__main__':
    data_out_path = sys.argv[1]
    download_ds = DownloadDatasets(out_path=data_out_path)

    taskIDs = [189354, 189356, 7593, 189355,
               7592, 34539, 168868, 14965, 146195, 146825, 168337, 168329, 146606, 168330, 167119, 3945, 168335, 9977,
               167120, 168338, 168332, 146212, 168331,
               146818, 10101, 146821, 168908, 9981, 31, 168909, 168910, 168911, 3917, 3, 12, 9952, 146822, 168912, 53]


    dataset_list = 'dataset_name,nrows,ncols,file_format,task_type,number_classes,original_url,target_feature,description\n'
    experiment_config = "--- \n \n"
    for tid in taskIDs:
        ds_name, target, nrows, ncols, dataset_id, original_data_url, task_type, number_classes= download_ds.download_dataset(taskID=tid)
        config_str = (f"- name: {ds_name} \n  dataset:\n     train: {{user}}\data\{ds_name}\{ds_name}_train.csv\n\
     test: {{user}}\data\{ds_name}\{ds_name}_test.csv\n     target: {target}\n  folds: 1\n\n")
        experiment_config += config_str
        dataset_list +=f'{ds_name},{nrows},{ncols},csv,{task_type},{number_classes},{original_data_url},{target}, "OpenML (TaskID={tid}; DatasetID={dataset_id})"\n'

    f = open(f'{data_out_path}/catdb_openml.yaml', 'w')
    f.write(experiment_config)

    f = open(f'{data_out_path}/dataset_list.csv', 'w')
    f.write(dataset_list)


