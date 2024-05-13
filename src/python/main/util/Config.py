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
from prompt.PromptChainTemplate import CatDBDataPreprocessingChainPrompt


class LLMSetting:
    def __init__(self):
        self.GPT_4_Limit = 8192
        self.GPT_4_1106_Preview_Limit = 4096
        self.GPT_3_5_Turbo_limit = 4096
        self.GPT_4_Turbo_Limit = 4096
        self.Llama3_70b_8192 = 8192

    def get_limit(self, model: str):
        if model == "skip":
            return -1

        if model == "gpt-4":
            return self.GPT_4_Limit

        elif model == "gpt-4-turbo":
            return self.GPT_4_Turbo_Limit

        elif model == "gpt-3.5-turbo":
            return self.GPT_3_5_Turbo_limit
        elif model =="llama3-70b-8192":
            return self.Llama3_70b_8192
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
               "CatDB": CatDBPrompt
               }
