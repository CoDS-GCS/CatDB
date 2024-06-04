from util import StaticValues


class BasicErrorPrompt(object):
    def __init__(self, pipeline_code: str, pipeline_error: str, schema_data: str,
                 data_source_train_path: str, data_source_test_path: str, *args, **kwargs):
        self.rules = []
        self.pipeline_code = pipeline_code
        self.pipeline_error = pipeline_error
        self.small_error_msg = None
        self.system_message_delimiter = None
        self.user_message_delimiter = None
        self.schema_data = schema_data
        self.data_source_train_path: data_source_train_path
        self.data_source_test_path: data_source_test_path

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
        prompt_items = [f"{_user_delimiter} {self.schema_data}\n", code, error, question]

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
        question = "Question: Modify the pipeline to return correct results."
        prompt_items = [f"{_user_delimiter} {code}", question]
        return f"\n\n{_user_delimiter}".join(prompt_items)

    def format_system_message(self):
        from util.Config import _system_delimiter
        rules = self.rules
        rules[0] = f"{_system_delimiter} {rules[0]}"
        return f"\n{_system_delimiter} ".join(rules)


class RuntimeErrorPrompt(BasicErrorPrompt):
    def __init__(self, evaluation_text: str, *args, **kwargs):
        BasicErrorPrompt.__init__(self, *args, **kwargs)
        self.rules = ['Task: You are expert in coding assistant. Your task is fix the error of this pipeline code.',
                      'Input: The user will provide a pipeline code enclosed in "<CODE> pipline code will be here. </CODE>", '
                      'and an error message enclosed in "<ERROR> error message will be here. </ERROR>".',
                      f"Rule 1: {StaticValues.rule_code_block}",
                      f"Rule 2: {evaluation_text}",
                      f'Rule 3 : {StaticValues.dp_rule_2.format(self.data_source_train_path, self.data_source_test_path)}'
                      ]

        min_length = min(len(self.pipeline_error), 2000)
        self.small_error_msg = self.pipeline_error[:min_length]


class ResultsErrorPrompt(BasicResultErrorPrompt):
    def __init__(self, evaluation_text: str, *args, **kwargs):
        BasicResultErrorPrompt.__init__(self, *args, **kwargs)
        self.rules = ['Task: You are expert in coding assistant. The following results did not achieved by direct '
                      'execution of the pipeline. Modify the code and return achievable results.'
                      'Your task is fix the error and return requested results of this pipeline code.',
                      'Input: The user will provide a pipeline code enclosed in "<CODE> pipline code will be here. </CODE>"',
                      f"Rule 1: {evaluation_text}",
                      f'Rule 2: {StaticValues.dp_rule_2.format(self.data_source_train_path, self.data_source_test_path)}'
                      ]