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
from prompt.PromptChainTemplate import DataPreprocessingChainPrompt
from prompt.PromptTemplate import AllPrompt

__GPT_4_Limit = 8192
__GPT_4_1106_Preview_Limit = 4096
__GPT_4_Turbo_Limit = 4096
__GPT_4o_Limit = 4096
__GPT_3_5_Turbo_limit = 4096
__Llama3_70b_8192 = 8192

_OPENAI = "OpenAI"
__GPT_system_delimiter = "### "
__GPT_user_delimiter = "### "

_META = "Meta"
__Llama_system_delimiter = "### "
__Llama_user_delimiter = "### "

_llm_model = None
_llm_platform = None
_system_delimiter = None
_user_delimiter = None
_max_token_limit = None


def set_config(model):
    global _llm_model
    global _llm_platform
    global _system_delimiter
    global _user_delimiter
    global _max_token_limit

    _llm_model = model

    if model == "gpt-4":
        _llm_platform = _OPENAI
        _max_token_limit = __GPT_4_Limit
        _user_delimiter = __GPT_user_delimiter
        _system_delimiter = __GPT_system_delimiter

    if model == "gpt-4-1106-preview_":
        _llm_platform = _OPENAI
        _max_token_limit = __GPT_4_1106_Preview_Limit
        _user_delimiter = __GPT_user_delimiter
        _system_delimiter = __GPT_system_delimiter

    elif model == "gpt-4-turbo":
        _llm_platform = _OPENAI
        _max_token_limit = __GPT_4_Turbo_Limit
        _user_delimiter = __GPT_user_delimiter
        _system_delimiter = __GPT_system_delimiter

    elif model == "gpt-4o":
        _max_token_limit = __GPT_4o_Limit
        _user_delimiter = __GPT_user_delimiter
        _system_delimiter = __GPT_system_delimiter

    elif model == "gpt-3.5-turbo":
        _llm_platform = _OPENAI
        _max_token_limit = __GPT_3_5_Turbo_limit
        _user_delimiter = __GPT_user_delimiter
        _system_delimiter = __GPT_system_delimiter

    elif model == "llama3-70b-8192":
        _llm_platform = _META
        _max_token_limit = __Llama3_70b_8192
        _user_delimiter = __Llama_user_delimiter
        _system_delimiter = __Llama_system_delimiter

    else:
        raise Exception(f"Model {model} is not implemented yet!")


CATEGORICAL_RATIO: float = 0.01
LOW_RATIO_THRESHOLD = 0.1
DISTINCT_THRESHOLD = 0.5

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
               "CatDBChain": DataPreprocessingChainPrompt
               }
