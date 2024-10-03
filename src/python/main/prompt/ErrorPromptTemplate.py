class BasicErrorPrompt(object):
    def __init__(self, pipeline_code: str, pipeline_error: str, schema_data: str = None, schema_dtypes: str= None, missing_value_rules: dict() = None, *args, **kwargs):
        self.rules = []
        self.pipeline_code = pipeline_code
        self.pipeline_error = pipeline_error
        self.missing_value_rules = missing_value_rules
        self.schema_dtypes = schema_dtypes
        self.small_error_msg = None
        self.system_message_delimiter = None
        self.user_message_delimiter = None
        self.schema_data = schema_data

    def format_prompt(self):
        return {
            "system_message": self.format_system_message(),
            "user_message": self.format_user_message()
        }

    def format_user_message(self):
        from util.Config import _user_delimiter
        code = f"<CODE>\n{self.pipeline_code}\n</CODE>"
        error = f"<ERROR>\n{self.small_error_msg}\n</ERROR>"
        question = ("Question: Fix the code error provided and return only the corrected pipeline without "
                    "additional explanations regarding the resolved error."
                    )
        prompt_items = []
        if "Input contains NaN" in self.small_error_msg and self.missing_value_rules is not None:
            prompt_items.append(f"{_user_delimiter} Explicitly do missing value imputation before column transfers in the prepreocessing section.")
            for k in self.missing_value_rules.keys():
                if self.missing_value_rules[k] is not None:
                    prompt_items.append(self.missing_value_rules[k])
        elif ("could not convert" in self.small_error_msg or "Encoders require their input argument must be uniformly strings or numbers" in self.small_error_msg) and self.schema_dtypes is not None:
            prompt_items.append(f"{_user_delimiter} Explicitly check the dataset column names and value types. Here is the column names and data types: \n {self.schema_dtypes}")
        elif "A given column is not a column of the dataframe" in self.small_error_msg:
            error = f"{error}\n{_user_delimiter} Remove target column from ColumnTransformer preprocessing part."
        else:
            prompt_items = [f"{_user_delimiter} {self.schema_data}\n"]
        prompt_items.append(code)
        prompt_items.append(error)
        prompt_items.append(question)
        return f"\n\n{_user_delimiter}".join(prompt_items)

    def format_system_message(self):
        from util.Config import _system_delimiter
        return f"{_system_delimiter}\n".join(self.rules)


class BasicResultErrorPrompt(object):
    def __init__(self, pipeline_code: str, *args, **kwargs):
        self.rules = []
        self.pipeline_code = pipeline_code
        self.system_message_delimiter = None
        self.user_message_delimiter = None

    def format_prompt(self):
        return {
            "system_message": self.format_system_message(),
            "user_message": self.format_user_message()
        }

    def format_user_message(self):
        from util.Config import _user_delimiter
        code = f"<CODE>\n{self.pipeline_code}\n</CODE>"
        question = f"Question: Modify the pipeline to return correct results.\n{_user_delimiter} don't use \"if __name__ == __main__:\" and replace it with a falt Python code."
        prompt_items = [f"{_user_delimiter} {code}", question]
        return f"\n\n{_user_delimiter}".join(prompt_items)

    def format_system_message(self):
        from util.Config import _system_delimiter
        rules = self.rules
        rules[0] = f"{_system_delimiter} {rules[0]}"
        return f"\n{_system_delimiter} ".join(rules)


class RuntimeErrorPrompt(BasicErrorPrompt):
    def __init__(self, evaluation_text: str, data_source_train_path: str, data_source_test_path: str, *args, **kwargs):
        from util.Config import _catdb_chain_DP_rules, _CODE_BLOCK
        BasicErrorPrompt.__init__(self, *args, **kwargs)
        self.rules = ['Task: You are expert in coding assistant. Your task is fix the error of this pipeline code.',
                      'Input: The user will provide a pipeline code enclosed in "<CODE> pipline code will be here. </CODE>", '
                      'and an error message enclosed in "<ERROR> error message will be here. </ERROR>".',
                      f"Rule 1: {_CODE_BLOCK}",
                      f"Rule 2: {evaluation_text}",
                      f'Rule 3 : {_catdb_chain_DP_rules["Rule_2"].format(data_source_train_path, data_source_test_path)}'
                      ]

        min_length = min(len(self.pipeline_error), 2000)
        self.small_error_msg = self.pipeline_error[:min_length]


class SyntaxErrorPrompt(BasicErrorPrompt):
    def __init__(self, *args, **kwargs):
        BasicErrorPrompt.__init__(self, *args, **kwargs)
        self.rules = ['Task: You are expert in coding assistant. Your task is fix the error of this pipeline code.',
                      'Input: The user will provide a pipeline code enclosed in "<CODE> pipline code will be here. </CODE>", '
                      'and an error message enclosed in "<ERROR> error message will be here. </ERROR>".']

        min_length = min(len(self.pipeline_error), 2000)
        self.small_error_msg = self.pipeline_error[:min_length]

    def format_user_message(self):
        from util.Config import _user_delimiter
        code = f"<CODE>\n{self.pipeline_code}\n</CODE>"
        error = f"<ERROR>\n{self.small_error_msg}\n</ERROR>"
        question = ("Question: Fix the code error provided and return only the corrected pipeline without "
                    "additional explanations regarding the resolved error."
                    )
        prompt_items = [code, error, question]

        return f"\n\n{_user_delimiter}".join(prompt_items)


class ResultsErrorPrompt(BasicResultErrorPrompt):
    def __init__(self, evaluation_text: str, data_source_train_path: str, data_source_test_path: str, *args, **kwargs):
        from util.Config import _catdb_chain_DP_rules
        BasicResultErrorPrompt.__init__(self, *args, **kwargs)
        self.rules = ['Task: You are expert in coding assistant. The following results did not achieved by direct '
                      'execution of the pipeline. Modify the code and return achievable results.'
                      'Your task is fix the error and return requested results of this pipeline code.',
                      'Input: The user will provide a pipeline code enclosed in "<CODE> pipline code will be here. </CODE>"',
                      f"Rule 1: {evaluation_text}",
                      f'Rule 2: {_catdb_chain_DP_rules["Rule_2"].format(data_source_train_path, data_source_test_path)}'
                      ]

