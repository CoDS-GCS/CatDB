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

from prompt.PromptChainTemplate import DataPreprocessingChainPrompt
from prompt.PromptChainTemplate import FeatureEngineeringChainPrompt
from prompt.PromptChainTemplate import ModelSelectionChainPrompt
import yaml

from .LLM_API_Key import LLM_API_Key

__gen_verify_mode = 'generate-and-verify'
__execute_mode = 'execute'

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
LOW_RATIO_THRESHOLD = 0.1
DISTINCT_THRESHOLD = 0.5

_OPENAI = "OpenAI"
_META = "Meta"
_GOOGLE = "Google"

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

_catdb_rules = dict()
_catdb_chain_DP_rules = dict()
_catdb_chain_FE_rules = dict()
_catdb_chain_MS_rules = dict()
_CODE_FORMATTING_IMPORT = None
_CODE_FORMATTING_PREPROCESSING = None
_CODE_FORMATTING_ADDING = None
_CODE_FORMATTING_DROPPING = None
_CODE_FORMATTING_TECHNIQUE = None
_CODE_FORMATTING_BINARY_EVALUATION = None
_CODE_FORMATTING_MULTICLASS_EVALUATION = None
_CODE_FORMATTING_REGRESSION_EVALUATION = None
_CODE_BLOCK = None
_CHAIN_RULE = None
_DATASET_DESCRIPTION = None


def load_config(system_log: str, llm_model: str = None, config_path: str = "Config.yaml",
                api_config_path: str = "APIKeys.yaml", rules_path: str = "Rules.yaml"):
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
    _system_log_file = system_log

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
        load_rules(rules_path=rules_path)


def load_rules(rules_path: str):
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
               "CatDBChainModelSelection": ModelSelectionChainPrompt
               }
