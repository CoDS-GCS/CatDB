from runcode.RunCode import RunCode


class DataCleaningPatch(object):
    def __init__(self, code: str, root_data, dataset_name, source_dataset_name, split_dataset: bool = False):
        self.code = code
        self.root_data = root_data
        self.dataset_name = f"{root_data}/{dataset_name}/{source_dataset_name}.csv"
        self.dataset_name_clean = f"{root_data}/{dataset_name}/{dataset_name}_clean.csv"
        self.dataset_name_train = f"{root_data}/{dataset_name}/{dataset_name}_train_clean.csv"
        self.dataset_name_test = f"{root_data}/{dataset_name}/{dataset_name}_test_clean.csv"
        self.dataset_name_verify = f"{root_data}/{dataset_name}/{dataset_name}_verify_clean.csv"
        self.split_dataset = split_dataset

    def apply_patch(self):
        src_patch = self.code
        if self.split_dataset:
            df_name = None
            for line in self.code.splitlines():
                if not line.startswith('#') and line.find(".to_csv"):
                    df_name = line.split(".to_csv")
                    df_name = df_name[0]
            package_src = "from sklearn.model_selection import train_test_split"
            split_src = f'data_train, data_test = train_test_split({df_name}, test_size=0.3, random_state=42)\n'
            split_src += f'_, data_verify = train_test_split(data_train, test_size=0.1, random_state=42)\n'
            split_src += f'data_train.to_csv("{self.dataset_name_train}", index=False)\n'
            split_src += f'data_test.to_csv("{self.dataset_name_test}", index=False)\n'
            split_src += f'data_verify.to_csv("{self.dataset_name_verify}", index=False)\n'

            src_patch = f"{package_src}\n{src_patch}\n{split_src}"

        src_patch = src_patch.replace("original_data.csv", self.dataset_name)
        src_patch = src_patch.replace("clean_data.csv", self.dataset_name_clean)
        RunCode.execute_code(src=src_patch, parse=None, run_mode="patch")
