class LLMSetting:
    def __init__(self):
        self.GPT_4_Token_Limit = 8192
        self.GPT_4_1106_Preview_Limit = 4096
        self.GPT_3_5_Turbo_limit = 4096

    def get_limit(self, model: str):
        if model == "skip":
            return -1

        if model == "gpt-4":
            return self.GPT_4_Token_Limit
        elif model == "gpt-3.5-turbo":
            return self.GPT_3_5_Turbo_limit
        else:
            raise Exception(f"Model {model} is not implemented yet!")


CATEGORICAL_RATIO: float = 0.01

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
REP_TYPE = {10000: "S",
            11000: "SDVC",
            10100: "SMVF",
            10010: "SSN",
            10001:"SCV",
            11100: "SDVMVF",
            10110:"SMVFSN",
            10101:"SMVFCV",
            10011:"SSNCV",
            11111:"ALL"}

PROMPT_FUNC = {"S": "SchemaPrompt",
               "SDVC": "SchemaDistinctValuePrompt",
               "SMVF": "SchemaMissingValueFrequencyPrompt",
               "SSN": "SchemaStatisticNumericPrompt",
               "SCV": "SchemaCategoricalValuesPrompt",
               "SDVCMVF": "SchemaDistinctValueCountMissingValueFrequencyPrompt",
               "SMVFSN":"SchemaMissingValueFrequencyStatisticNumericPrompt",
               "SMVFCV": "SchemaMissingValueFrequencyCategoricalValuesPrompt",
               "SSNCV": "SchemaStatisticNumericCategoricalValuesPrompt",
               "ALL": "AllPrompt"}



