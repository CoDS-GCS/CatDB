import pandas as pd
import os

from prompt.PromptTemplate import SchemaPrompt
from prompt.PromptTemplate import SchemaDistinctValuePrompt
from prompt.PromptTemplate import SchemaMissingValueFrequencyPrompt
from prompt.PromptTemplate import SchemaStatisticNumericPrompt
from prompt.PromptTemplate import SchemaCategoricalValuesPrompt
from prompt.PromptTemplate import SchemaDistinctValueCountMissingValueFrequencyPrompt
from prompt.PromptTemplate import SchemaDistinctValueCountStatisticNumericPrompt
from prompt.PromptTemplate import SchemaMissingValueFrequencyStatisticNumericPrompt
from prompt.PromptTemplate import SchemaMissingValueFrequencyCategoricalValuesPrompt
from prompt.PromptTemplate import SchemaStatisticNumericCategoricalValuesPrompt
from prompt.PromptTemplate import CatDBPrompt
from prompt.PromptTemplate import AllPrompt
from prompt.PromptTemplateMultiTable import CatDBMultiTablePrompt

from prompt.PromptChainTemplate import DataPreprocessingChainPrompt
from prompt.PromptChainTemplate import FeatureEngineeringChainPrompt
from prompt.PromptChainTemplate import ModelSelectionChainPrompt

from prompt.PromptTemplateMissing import CatDBMissingValuePrompt
import yaml

from .LLM_API_Key import LLM_API_Key

__gen_verify_mode = 'generate-and-verify'
__execute_mode = 'execute'
_data_cleaning_mode = 'data-cleaning'

Google_SCOPES = ['https://www.googleapis.com/auth/cloud-platform',
                 'https://www.googleapis.com/auth/generative-language.tuning']
_google_token_file_path = None
_google_client_secret_file_path = None
_fintune_train_data = None
_fintune_train_data_target_attribute = None

_missing_value_train_data = None
_missing_value_train_data_target_attribute = None
_missing_value_train_data_samples = None

__sub_task_data_preprocessing = "DataPreprocessing"
__sub_task_feature_engineering = "FeatureEngineering"
__sub_task_model_selection = "ModelSelection"

default_max_token_limit = 4096
default_max_output_tokens = 8192
default_delay = 0
default_temperature = 0
default_top_p = 0.95
default_top_k = 64
default_system_delimiter = "### "
default_user_delimiter = "### "

CATEGORICAL_RATIO: float = 0.01
LOW_RATIO_THRESHOLD = 0.05
DISTINCT_THRESHOLD = 0.5
CATEGORICAL_VALUES_RESTRICTED_SIZE = 50

_OPENAI = "OpenAI"
_META = "Meta"
_GOOGLE = "Google"
_DEEPSEEK = "DeepSeek"

_llm_model = None
_llm_platform = None
_system_delimiter = None
_user_delimiter = None
_max_token_limit = None
_max_out_token_limit = None
_delay = None
_temperature = None
_top_p = None
_top_k = None
_last_API_Key = None
_LLM_API_Key = None
_system_log_file = None
_enable_cache = False

_catdb_rules = dict()
_catdb_chain_DP_rules = dict()
_catdb_chain_FE_rules = dict()
_catdb_chain_MS_rules = dict()
_catdb_categorical_data_cleaning_rules = dict()
_catdb_categorical_catalog_cleaning_rules = dict()

_CODE_FORMATTING_IMPORT = None
_CODE_FORMATTING_PREPROCESSING = None
_CODE_FORMATTING_ADDING = None
_CODE_FORMATTING_DROPPING = None
_CODE_FORMATTING_TECHNIQUE = None
_CODE_FORMATTING_BINARY_EVALUATION = None
_CODE_FORMATTING_MULTICLASS_EVALUATION = None
_CODE_FORMATTING_REGRESSION_EVALUATION = None
_CODE_FORMATTING_ACC_EVALUATION = None
_CODE_BLOCK = None
_CHAIN_RULE = None
_DATASET_DESCRIPTION = None


def load_config(system_log: str, llm_model: str = None, config_path: str = "Config.yaml",
                api_config_path: str = None, rules_path: str = "Rules.yaml",
                data_cleaning_rules_path="RulesDataCleaning.yaml", evaluation_acc: bool = False, enable_cache: bool= False):
    if api_config_path is None:
        api_config_path = os.environ.get("APIKeys_File")
    global _llm_model
    global _llm_platform
    global _system_delimiter
    global _user_delimiter
    global _max_token_limit
    global _max_out_token_limit
    global _delay
    global _last_API_Key
    global _LLM_API_Key
    global _system_log_file
    global _temperature
    global _top_k
    global _top_p
    global _enable_cache
    _system_log_file = system_log
    _enable_cache = enable_cache
    with open(config_path, "r") as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            for conf in configs:
                plt = conf.get("llm_platform")
                try:
                    if conf.get(llm_model) is not None:
                        _llm_model = llm_model
                        _llm_platform = plt

                        try:
                            _system_delimiter = conf.get(llm_model).get('system_delimiter')
                        except:
                            _system_delimiter = default_system_delimiter

                        try:
                            _user_delimiter = conf.get(llm_model).get('user_delimiter')
                        except:
                            _user_delimiter = default_user_delimiter

                        try:
                            _max_token_limit = int(conf.get(llm_model).get('token_limit'))
                        except:
                            _max_token_limit = default_max_token_limit

                        try:
                            _max_out_token_limit = int(conf.get(llm_model).get('max_output_tokens'))
                        except:
                            _max_out_token_limit = default_max_output_tokens

                        try:
                            _delay = int(conf.get(llm_model).get('delay'))
                        except:
                            _delay = default_delay

                        try:
                            _temperature = float(conf.get(llm_model).get('temperature'))
                        except:
                            _temperature = default_temperature

                        try:
                            _top_k = int(conf.get(llm_model).get('top_k'))
                        except:
                            _top_k = default_top_k

                        try:
                            _top_p = float(conf.get(llm_model).get('top_p'))
                        except:
                            _top_p = default_top_p

                        break
                except Exception as ex:
                    pass

        except yaml.YAMLError as ex:
            raise Exception(ex)

        if _llm_model is None:
            raise Exception(f'Error: model "{llm_model}" is not in the Config.yaml list!')

        _LLM_API_Key = LLM_API_Key(api_config_path=api_config_path)
        load_rules(rules_path=rules_path, evaluation_acc=evaluation_acc)
        load_data_cleaning_rules(rules_path=data_cleaning_rules_path)


def load_rules(rules_path: str, evaluation_acc: bool = False):
    global _catdb_rules
    global _catdb_chain_DP_rules
    global _catdb_chain_FE_rules
    global _catdb_chain_MS_rules
    global _CODE_FORMATTING_IMPORT
    global _CODE_FORMATTING_PREPROCESSING
    global _CODE_FORMATTING_ADDING
    global _CODE_FORMATTING_DROPPING
    global _CODE_FORMATTING_TECHNIQUE
    global _CODE_FORMATTING_BINARY_EVALUATION
    global _CODE_FORMATTING_MULTICLASS_EVALUATION
    global _CODE_FORMATTING_REGRESSION_EVALUATION
    global _CODE_FORMATTING_ACC_EVALUATION
    global _CODE_BLOCK
    global _CHAIN_RULE
    global _DATASET_DESCRIPTION

    with (open(rules_path, "r") as f):
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            for conf in configs:
                plt = conf.get("Config")
                if plt != 'CodeFormat':
                    rls = dict()
                    for k, v in conf.items():
                        rls[k] = v
                        if plt == "CatDB":
                            _catdb_rules = rls
                        elif plt == "CatDBChainDP":
                            _catdb_chain_DP_rules = rls
                        elif plt == "CatDBChainFE":
                            _catdb_chain_FE_rules = rls
                        elif plt == "CatDBChainMS":
                            _catdb_chain_MS_rules = rls

                else:
                    for k, v in conf.items():
                        if k == 'CODE_FORMATTING_IMPORT':
                            _CODE_FORMATTING_IMPORT = v
                        elif k == 'CODE_FORMATTING_PREPROCESSING':
                            _CODE_FORMATTING_PREPROCESSING = v
                        elif k == 'CODE_FORMATTING_ADDING':
                            _CODE_FORMATTING_ADDING = v
                        elif k == 'CODE_FORMATTING_DROPPING':
                            _CODE_FORMATTING_DROPPING = v
                        elif k == 'CODE_FORMATTING_TECHNIQUE':
                            _CODE_FORMATTING_TECHNIQUE = v
                        elif k == 'CODE_FORMATTING_BINARY_EVALUATION':
                            _CODE_FORMATTING_BINARY_EVALUATION = v
                        elif k == 'CODE_FORMATTING_ACC_EVALUATION':
                            _CODE_FORMATTING_ACC_EVALUATION = v
                        elif k == 'CODE_FORMATTING_MULTICLASS_EVALUATION':
                            _CODE_FORMATTING_MULTICLASS_EVALUATION = v
                        elif k == 'CODE_FORMATTING_REGRESSION_EVALUATION':
                            _CODE_FORMATTING_REGRESSION_EVALUATION = v
                        elif k == 'CODE_BLOCK':
                            _CODE_BLOCK = v
                        elif k == 'CHAIN_RULE':
                            _CHAIN_RULE = v
                        elif k == 'DATASET_DESCRIPTION':
                            _DATASET_DESCRIPTION = v
        except yaml.YAMLError as ex:
            raise Exception(ex)

    if evaluation_acc:
        _CODE_FORMATTING_BINARY_EVALUATION = _CODE_FORMATTING_ACC_EVALUATION
        _CODE_FORMATTING_MULTICLASS_EVALUATION = _CODE_FORMATTING_ACC_EVALUATION


def load_data_cleaning_rules(rules_path: str):
    global _catdb_categorical_data_cleaning_rules
    global _catdb_categorical_catalog_cleaning_rules

    with (open(rules_path, "r") as f):
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            for conf in configs:
                plt = conf.get("Config")
                categorical_rls = dict()
                for k, v in conf.get("Categorical").items():
                    categorical_rls[k] = v
                if plt == "CatDB":
                    _catdb_categorical_data_cleaning_rules = categorical_rls
                elif plt == "Type-Infer":
                    _catdb_categorical_catalog_cleaning_rules = categorical_rls
        except yaml.YAMLError as ex:
            raise Exception(ex)


def convert_df_to_string(df, row_prefix: str = None):
    indexes = df.index.tolist()
    x = df.to_string(header=True, index=False, index_names=False).split('\n')
    xl = [','.join(ele.split()) for ele in x]
    if row_prefix is not None:
        nxl = []
        for index, val in enumerate(xl):
            if index > 0:
                nxl.append(f"{row_prefix} {indexes[index - 1]}: {val}")
            else:
                nxl.append(f"### Header: {val}")
        xl = nxl
    return "\n".join(xl)


def load_missing_value_dataset(data, target_attribute: str = None, task_type: str = None, number_samples: int = 0):
    global _missing_value_train_data
    global _missing_value_train_data_target_attribute
    global _missing_value_train_data_samples

    _missing_value_train_data_target_attribute = target_attribute

    df = data.dropna(how='any', axis=0)
    if task_type in {'binary', 'multiclass'}:
        df = df.groupby(target_attribute).sample(number_samples, replace=True)
        _missing_value_train_data_samples = len(df)
        # _missing_value_train_data = convert_df_to_string(df)
        # classes = df[target_attribute].unique()
        # _missing_value_train_data = dict()
        # for c in classes:
        #     tmp_df = df.loc[df[target_attribute] == c]
        #     tmp_df = tmp_df.sample(min(number_samples, len(tmp_df)), replace=True)
        #     _missing_value_train_data[c] = convert_df_to_string(tmp_df)
        #     tmp_df.to_csv(f"/home/saeed/Downloads/tmp/{c}.csv")
    else:
        df = df.sample(number_samples, replace=True)
        _missing_value_train_data_samples = len(df)
    _missing_value_train_data = convert_df_to_string(df=df)


def set_finetune_file_path(google_client_secret_file_path: str = None, google_token_file_path: str = None,
                           dataset_path: str = None, target_attribute: str = None, task_type: str = None):
    global _google_token_file_path
    global _google_client_secret_file_path
    global _fintune_train_data
    global _fintune_train_data_target_attribute

    _google_token_file_path = google_token_file_path
    _google_client_secret_file_path = google_client_secret_file_path
    _fintune_train_data_target_attribute = target_attribute

    df = pd.read_csv(dataset_path)
    if task_type in {'binary', 'multiclass'}:
        _fintune_train_data = df.groupby(target_attribute).sample(100, replace=True)
    else:
        _fintune_train_data = df.sample(1000, replace=True)


# 10000 : Schema = S
# 11000 : Schema + Distinct Value Count = S + DVC = SDVC
# 10100 : Schema + Missing Value Frequency + S + MVF = SMVF
# 10010 : Schema + Statistical Number = S + SN = SSN
# 10001 : Schema + Categorical Values = S + CV = SCV
# 11100 : Schema + Distinct Values Count + Missing Value Frequency = S + DV + MVF = SDVCMVF
# 10110 : Schema + Missing Value Frequency + Statistical Number = S + MVF + FN = SMVFSN
# 10101 : Schema + Missing Value Frequency + Categorical Values = S + MVF + CV = SMVFCV
# 10011 : Schema + Statistical Number + Categorical Values = S + SN + CV = SSNCV
# 11111 : Schema + Distinct Value Count + Missing Value Frequency + Statistical Number + Categorical Values = ALL

PROFILE_TYPE = {"S": 10000, "DVC": 1000, "MVF": 100, "SN": 10, "CV": 1}
PROFILE_TYPE_VAL = [10000, 1000, 100, 10, 1]
REP_TYPE = {10000: "S",
            11000: "SDVC",
            10100: "SMVF",
            10010: "SSN",
            10001: "SCV",
            11100: "SDVCMVF",
            11010: "SDVCSN",
            10110: "SMVFSN",
            10101: "SMVFCV",
            10011: "SSNCV",
            11111: "ALL"}

PROMPT_FUNC = {"S": SchemaPrompt,
               "SDVC": SchemaDistinctValuePrompt,
               "SMVF": SchemaMissingValueFrequencyPrompt,
               "SSN": SchemaStatisticNumericPrompt,
               "SCV": SchemaCategoricalValuesPrompt,
               "SDVCMVF": SchemaDistinctValueCountMissingValueFrequencyPrompt,
               "SDVCSN": SchemaDistinctValueCountStatisticNumericPrompt,
               "SMVFSN": SchemaMissingValueFrequencyStatisticNumericPrompt,
               "SMVFCV": SchemaMissingValueFrequencyCategoricalValuesPrompt,
               "SSNCV": SchemaStatisticNumericCategoricalValuesPrompt,
               "ALL": AllPrompt,
               "CatDB": CatDBPrompt,
               "CatDBChainDataPreprocessing": DataPreprocessingChainPrompt,
               "CatDBChainFeatureEngineering": FeatureEngineeringChainPrompt,
               "CatDBChainModelSelection": ModelSelectionChainPrompt,
               "CatDBMissingValue": CatDBMissingValuePrompt
               }

PROMPT_FUNC_MULTI_TABLE = {
    "CatDB": CatDBMultiTablePrompt,
    "CatDBChainDataPreprocessing": DataPreprocessingChainPrompt,
    "CatDBChainFeatureEngineering": FeatureEngineeringChainPrompt,
    "CatDBChainModelSelection": ModelSelectionChainPrompt,
    "CatDBMissingValue": CatDBMissingValuePrompt
}
