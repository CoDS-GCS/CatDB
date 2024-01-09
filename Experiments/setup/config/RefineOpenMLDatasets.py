import sys
import pandas as pd


class RefineDataset(object):
    def __init__(self, out_path):
        self.out_path = out_path

    def load_dataset(self, ds_name: str, ncols: int, train_path: str, test_path: str, val_path: str, targe_path: str):
        col_names = [f'col_{i}' for i in range(0, ncols)]
        ds_train = pd.read_csv(train_path, sep=' ', header=None, names=col_names, low_memory=False,
                               lineterminator='\n', index_col=False)
        ds_test = pd.read_csv(test_path, sep=' ', header=None, names=col_names, low_memory=False,
                              lineterminator='\n', index_col=False)
        ds_val = pd.read_csv(val_path, sep=' ', header=None, names=col_names, low_memory=False,
                             lineterminator='\n', index_col=False)
        ds_target = pd.read_csv(targe_path, sep=' ', header=None, index_col=False)

        (target_nrows, target_ncols) = ds_target.shape
        ds_target.columns = [f'target_{i}' for i in range(0, target_ncols)]

        if target_ncols == 1:
            ds_target_new = ds_target
        else:
            ds_target_new = pd.DataFrame(columns=['target_0'])
            for r in range(0, target_nrows):
                for c in range(0, target_ncols):
                    class_value = ds_target.iat[r, c]
                    if class_value == 1:
                        ds_target_new.loc[r] = [c]

        ds_train = pd.concat([ds_train.reset_index(drop=True), ds_target_new], axis=1).dropna(how= "all", axis=1)
        ds_test = ds_test.dropna(how= "all", axis=1)
        ds_test["target_0"] = -99999
        ds_val = ds_val.dropna(how= "all", axis=1)
        ds_val["target_0"] = -99999

        ds_train.to_csv(f'{self.out_path}/{ds_name}_train.csv', index=False)
        ds_test.to_csv(f'{self.out_path}/{ds_name}_test.csv', index=False)
        ds_val.to_csv(f'{self.out_path}/{ds_name}_val.csv', index=False)

        return 'target_0'


if __name__ == '__main__':
    data_source = "/home/saeed/Documents/Github/CatDB/Experiments/data" #sys.argv[1]
    ds_name = "sylvine" #sys.argv[2]

    train_path = f'{data_source}/{ds_name}/{ds_name}_train.data'
    test_path = f'{data_source}/{ds_name}/{ds_name}_test.data'
    val_path = f'{data_source}/{ds_name}/{ds_name}_valid.data'
    target_path = f'{data_source}/{ds_name}/{ds_name}_train.solution'

    public_info = f'{data_source}/{ds_name}/{ds_name}_public.info'
    ncols = 0
    with open(public_info, 'r') as file:
        lines = file.readlines()
        for line in lines:
            key_value = line.strip().split("=")
            if key_value[0].strip() == "feat_num":
                ncols = int(key_value[1].strip())
                break

    rd = RefineDataset(out_path=f'{data_source}/{ds_name}')
    target = rd.load_dataset(ds_name=ds_name, ncols=ncols, train_path=train_path, test_path=test_path,
                             val_path=val_path, targe_path=target_path)

    if target is not None:
        config_strs = [f"- name: {ds_name}",
                       "  dataset:",
                       f"    train: \'{{user}}/data/{ds_name}/{ds_name}_train.csv\'",
                       f"    test: \'{{user}}/data/{ds_name}/{ds_name}_test.csv\'",
                       f"    target: {target}",
                       "  folds: 1",
                       "\n"]
        config_str = "\n".join(config_strs)

        yaml_file = f'{data_source}/{ds_name}/{ds_name}.yaml'
        f = open(yaml_file, 'w')
        f.write("--- \n \n")
        f.write(config_str)
        f.close()
