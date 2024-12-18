from argparse import ArgumentParser
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from metagpt.ext.sela.insights.solution_designer import SolutionDesigner
from metagpt.ext.sela.utils import DATA_CONFIG
from runner.LogResults import LogDataPrepare
from runner.LogResults import LogDataPrepare
import time

BASE_USER_REQUIREMENT = """
This is a {datasetname} dataset. Your goal is to predict the target column `{target_col}`.
Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. 
Report {metric} on the Train and Test data. Do not plot or make any visualizations.
"""

USE_AG = """
- Please use autogluon for model training with presets='medium_quality', time_limit=None, give dev dataset to tuning_data, and use right eval_metric.
"""

TEXT_MODALITY = """
- You could use models from transformers library for this text dataset.
- Use gpu if available for faster training.
"""

IMAGE_MODALITY = """
- You could use models from transformers/torchvision library for this image dataset.
- Use gpu if available for faster training.
"""

STACKING = """
- To avoid overfitting, train a weighted ensemble model such as StackingClassifier or StackingRegressor.
- You could do some quick model prototyping to see which models work best and then use them in the ensemble. 
"""


SPECIAL_INSTRUCTIONS = {"ag": USE_AG, "stacking": STACKING, "text": TEXT_MODALITY, "image": IMAGE_MODALITY}

DI_INSTRUCTION = """
## Attention
1. Please do not leak the target label in any form during training.
2. Test set does not have the target column.
3. When conducting data exploration or analysis, don't print out the results of your findings.
4. You should perform transformations on train, dev, and test sets at the same time (it's a good idea to define functions for this and avoid code repetition).
5. When scaling or transforming features, make sure the target column is not included.
6. You could utilize dev set to validate and improve model training. {special_instruction}

## Saving Train, Dev and Test Predictions
1. Save the prediction results of the train set, dev set and test set in `train_predictions.csv`, `dev_predictions.csv` and `test_predictions.csv` respectively in the output directory. 
- All files should contain a single column named `target` with the predicted values.
2. Make sure the prediction results are in the same format as the target column in the original training set. 
- For instance, if the original target column is a list of string, the prediction results should also be strings.

## Output Performance
Print the Train, Test and Dev set performance in the last step.

# Output dir
{output_dir}
"""

TASK_PROMPT = """
# User requirement
{user_requirement}
{additional_instruction}
# Data dir
train set (with labels): {train_path}
dev set (with labels): {dev_path}
test set (without labels): {test_path}
dataset description: {data_info_path} (During EDA, you can use this file to get additional information about the dataset)
"""

def get_split_dataset_path(dataset_name, config):
    datasets_dir = config["datasets_dir"]
    if dataset_name in config["datasets"]:
        dataset = config["datasets"][dataset_name]
        data_path = os.path.join(datasets_dir, dataset["dataset"])
        split_datasets = {
            "train": os.path.join(data_path, "split_train.csv"),
            "dev": os.path.join(data_path, "split_dev.csv"),
            "dev_wo_target": os.path.join(data_path, "split_dev_wo_target.csv"),
            "dev_target": os.path.join(data_path, "split_dev_target.csv"),
            "test": os.path.join(data_path, "split_test.csv"),
            "test_wo_target": os.path.join(data_path, "split_test_wo_target.csv"),
            "test_target": os.path.join(data_path, "split_test_target.csv"),
        }
        return split_datasets
    else:
        raise ValueError(
            f"Dataset {dataset_name} not found in config file. Available datasets: {config['datasets'].keys()}"
        )


def get_user_requirement(task_name, config):
    # datasets_dir = config["datasets_dir"]
    if task_name in config["datasets"]:
        dataset = config["datasets"][task_name]
        # data_path = os.path.join(datasets_dir, dataset["dataset"])
        user_requirement = dataset["user_requirement"]
        return user_requirement
    else:
        raise ValueError(
            f"Dataset {task_name} not found in config file. Available datasets: {config['datasets'].keys()}"
        )


def save_datasets_dict_to_yaml(datasets_dict, name="datasets.yaml"):
    with open(name, "w") as file:
        yaml.dump(datasets_dict, file)


def create_dataset_dict(dataset):
    dataset_dict = {
        "dataset": dataset.name,
        "user_requirement": dataset.create_base_requirement(),
        "metric": dataset.get_metric(),
        "target_col": dataset.target_col,
    }
    return dataset_dict


def generate_di_instruction(output_dir, special_instruction):
    if special_instruction:
        special_instruction_prompt = SPECIAL_INSTRUCTIONS[special_instruction]
    else:
        special_instruction_prompt = ""
    additional_instruction = DI_INSTRUCTION.format(
        output_dir=output_dir, special_instruction=special_instruction_prompt
    )
    return additional_instruction


def generate_task_requirement(task_name, data_config, is_di=True, special_instruction=None):
    user_requirement = get_user_requirement(task_name, data_config)
    split_dataset_path = get_split_dataset_path(task_name, data_config)
    train_path = split_dataset_path["train"]
    dev_path = split_dataset_path["dev"]
    test_path = split_dataset_path["test_wo_target"]
    work_dir = data_config["work_dir"]
    output_dir = f"{work_dir}/{task_name}"
    datasets_dir = data_config["datasets_dir"]
    data_info_path = f"{datasets_dir}/{task_name}/dataset_info.json"
    if is_di:
        additional_instruction = generate_di_instruction(output_dir, special_instruction)
    else:
        additional_instruction = ""
    user_requirement = TASK_PROMPT.format(
        user_requirement=user_requirement,
        train_path=train_path,
        dev_path=dev_path,
        test_path=test_path,
        additional_instruction=additional_instruction,
        data_info_path=data_info_path,
    )
    #print(user_requirement)
    return user_requirement


class ExpDataset:
    description: str = None
    metadata: dict = None
    dataset_dir: str = None
    target_col: str = None
    name: str = None
    data_source_path: str = None
    data_source_train_path: str = None
    data_source_test_path: str = None
    data_source_verify_path: str = None
    task_type = None

    def __init__(self, name, dataset_dir, **kwargs):
        self.name = name
        self.dataset_dir = dataset_dir
        self.target_col = kwargs.get("target_col", None)
        self.force_update = kwargs.get("force_update", False)
        self.data_source_path = kwargs.get("data_source_path", None)
        self.data_source_train_path = kwargs.get("data_source_train_path", None)
        self.data_source_test_path = kwargs.get("data_source_test_path", None)
        self.data_source_verify_path = kwargs.get("data_source_verify_path", None)
        self.task_type = kwargs.get("task_type", None)

        if not os.path.exists(f"{self.dataset_dir}/{self.name}"):
            os.makedirs(f"{self.dataset_dir}/{self.name}")

        # if not os.path.exists(f"{self.dataset_dir}/{self.name}/raw"):
        #     os.makedirs(f"{self.dataset_dir}/{self.name}/raw")    

        self.save_dataset(target_col=self.target_col)

    def check_dataset_exists(self):
        fnames = [
            "split_train.csv",
            "split_dev.csv",
            "split_test.csv",
            "split_dev_wo_target.csv",
            "split_dev_target.csv",
            "split_test_wo_target.csv",
            "split_test_target.csv",
        ]
        for fname in fnames:
            if not os.path.exists(Path(self.dataset_dir, self.name, fname)):
                return False
        return True

    def check_datasetinfo_exists(self):
        return os.path.exists(Path(self.dataset_dir, self.name, "dataset_info.json"))

    def get_raw_dataset(self):        
        train_df = pd.read_csv(self.data_source_train_path)
        test_df = pd.read_csv(self.data_source_test_path)
        dev_df = pd.read_csv(self.data_source_verify_path)
        
        return train_df, test_df, dev_df

    def get_dataset_info(self):
        raw_df = pd.read_csv(self.data_source_path)
        metadata = {
            "NumberOfClasses": raw_df[self.target_col].nunique(),
            "NumberOfFeatures": raw_df.shape[1],
            "NumberOfInstances": raw_df.shape[0],
            "NumberOfInstancesWithMissingValues": int(raw_df.isnull().any(axis=1).sum()),
            "NumberOfMissingValues": int(raw_df.isnull().sum().sum()),
            "NumberOfNumericFeatures": raw_df.select_dtypes(include=["number"]).shape[1],
            "NumberOfSymbolicFeatures": raw_df.select_dtypes(include=["object"]).shape[1],
        }

        df_head_text = self.get_df_head(raw_df)

        dataset_info = {
            "name": self.name,
            "description": "",
            "target_col": self.target_col,
            "metadata": metadata,
            "df_head": df_head_text,
        }
        return dataset_info

    def get_df_head(self, raw_df):
        return raw_df.head().to_string(index=False)

    def get_metric(self):
        if self.task_type == "binary":
            metric = "auc"
        elif self.task_type == "multiclass":
            metric = "auc-ovr"
        elif self.task_type == "regression":
            metric = "r2"

        return metric

    def create_base_requirement(self):
        metric = self.get_metric()
        req = BASE_USER_REQUIREMENT.format(datasetname=self.name, target_col=self.target_col, metric=metric)
        return req

    def save_dataset(self, target_col):
        train_df, test_df, dev_df = self.get_raw_dataset()

        if not self.check_dataset_exists() or self.force_update:
            print(f"Saving Dataset {self.name} in {self.dataset_dir}")
            self.save_raw_dataset(train_df=train_df, test_df=test_df, dev_df=dev_df, target_col=target_col)
        else:
            print(f"Dataset {self.name} already exists")
        if not self.check_datasetinfo_exists() or self.force_update:
            print(f"Saving Dataset info for {self.name}")
            dataset_info = self.get_dataset_info()
            self.save_datasetinfo(dataset_info)
        else:
            print(f"Dataset info for {self.name} already exists")

    def save_datasetinfo(self, dataset_info):
        with open(Path(self.dataset_dir, self.name, "dataset_info.json"), "w", encoding="utf-8") as file:
            # utf-8 encoding is required
            json.dump(dataset_info, file, indent=4, ensure_ascii=False)

    def save_split_datasets(self, df, split, target_col=None):
        path = Path(self.dataset_dir, self.name)
        df.to_csv(Path(path, f"split_{split}.csv"), index=False)
        if target_col:
            df_wo_target = df.drop(columns=[target_col])
            df_wo_target.to_csv(Path(path, f"split_{split}_wo_target.csv"), index=False)
            df_target = df[[target_col]].copy()
            if target_col != "target":
                df_target["target"] = df_target[target_col]
                df_target = df_target.drop(columns=[target_col])
            df_target.to_csv(Path(path, f"split_{split}_target.csv"), index=False)

    def save_raw_dataset(self, train_df, test_df, dev_df, target_col):
        if not target_col:
            raise ValueError("Target column not provided")
        
        self.save_split_datasets(train_df, "train")
        self.save_split_datasets(dev_df, "dev", target_col)
        self.save_split_datasets(test_df, "test", target_col)


async def process_dataset(dataset, solution_designer: SolutionDesigner, save_analysis_pool, datasets_dict, output_path, llm_model):
    time_start = time.time()
     
    prompt_tokens = 0
    completion_tokens = 0
    if save_analysis_pool:
        _, prompt_tokens, completion_tokens = await solution_designer.generate_solutions(dataset.get_dataset_info(), dataset.name)
    dataset_dict = create_dataset_dict(dataset)
    datasets_dict["datasets"][dataset.name] = dataset_dict
 
    time_end = time.time()
    log_result = LogDataPrepare(dataset_name=dataset.name, 
                                sub_task="ProcessDataset", 
                                llm_model=llm_model, 
                                time_total=time_end - time_start,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                all_tokens_count= prompt_tokens + completion_tokens)
    
    log_result.save_results(result_output_path=output_path)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--root-data-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default="gemini-1.5-pro-latest")    
    parser.add_argument('--output-path', type=str, default="/tmp/results.csv")
    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    if args.root_data_path is None:
        raise Exception("--root-data-path is a required parameter!")


    # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            try:
                args.data_source_path = f"{args.root_data_path}/{args.dataset_name}/{args.dataset_name}.csv"
                args.data_source_train_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('train')}"
                args.data_source_test_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('test')}"
                args.data_source_verify_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('verify')}"
                
            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)
    args.force_update = True    
    args.save_analysis_pool = True

    return args


if __name__ == "__main__":
    args = parse_args()
    datasets_dir = DATA_CONFIG["datasets_dir"]
    force_update = args.force_update
    save_analysis_pool = args.save_analysis_pool
    datasets_dict = {"datasets": {}}
    solution_designer = SolutionDesigner()
    dataset_name = args.dataset_name
    target_col = args.target_attribute

    custom_dataset = ExpDataset(dataset_name, datasets_dir, target_col=target_col, force_update=force_update, task_type=args.task_type,
                                data_source_path=args.data_source_path, data_source_train_path=args.data_source_train_path,
                                data_source_test_path=args.data_source_test_path, data_source_verify_path=args.data_source_verify_path)

    asyncio.run(process_dataset(custom_dataset, solution_designer, save_analysis_pool, datasets_dict, output_path=args.output_path, llm_model = args.llm_model))
    save_datasets_dict_to_yaml(datasets_dict)