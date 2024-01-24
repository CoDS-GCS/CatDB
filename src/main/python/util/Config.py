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


