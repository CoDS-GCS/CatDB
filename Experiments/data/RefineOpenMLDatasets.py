import sys
import pandas as pd


class RefineDataset(object):
    def __init__(self, out_path):
        self.out_path = out_path

    def load_dataset(self, ds_name: str, ncols:int, train_path: str, test_path: str, val_path: str, targe_path: str):
        col_names = [f'col_{i}' for i in range(0, ncols)]
        ds_train = pd.read_csv(train_path, delimiter=' ', header=None, names=col_names, low_memory=False, lineterminator='\n')
        ds_test = pd.read_csv(test_path, delimiter=' ', header=None, names=col_names, low_memory=False, lineterminator='\n')
        ds_val = pd.read_csv(val_path, delimiter=' ', header=None, names=col_names, low_memory=False, lineterminator='\n')
        ds_target = pd.read_csv(targe_path, delimiter=' ', header=None)

        target_cols = [f'target_{i}' for i in range(0, ds_target.shape[1])]
        ds_target.set_axis(target_cols, axis='columns', copy=False)
        ds_train = pd.concat([ds_train.reset_index(drop=True), ds_target], axis=1)

        ds_train.to_csv(f'{self.out_path}/{ds_name}_train.csv', index=False)
        ds_test.to_csv(f'{self.out_path}/{ds_name}_test.csv', index=False)
        ds_val.to_csv(f'{self.out_path}/{ds_name}_val.csv', index=False)

        if ds_target.shape[1] == 1:
            return 'target_0'
        else:
            return None


if __name__ == '__main__':
    data_source = sys.argv[1]
    ds_name = sys.argv[2]

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
    target = rd.load_dataset(ds_name=ds_name, ncols=ncols, train_path=train_path, test_path=test_path, val_path=val_path, targe_path=target_path)

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