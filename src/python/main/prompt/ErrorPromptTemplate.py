from util import StaticValues


class BasicErrorPrompt(object):
    def __init__(self, pipeline_code: str, pipeline_error: str, *args, **kwargs):
        self.rules = []
        self.pipeline_code = pipeline_code
        self.pipeline_error = pipeline_error
        self.small_error_msg = None
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
        error = f"<ERROR>\n{self.small_error_msg}\n</ERROR>"
        question = ("Question: Fix the code error provided and return only the corrected pipeline without "
                    "additional explanations regarding the resolved error."
                    )
        prompt_items = [code, error, question]

        return f"{_user_delimiter}".join(prompt_items)

    def format_system_message(self):
        from util.Config import _system_delimiter
        return f"{_system_delimiter}".join(self.rules)


class RuntimeErrorPrompt(BasicErrorPrompt):
    def __init__(self, *args, **kwargs):
        BasicErrorPrompt.__init__(self, *args, **kwargs)
        self.rules = ['Task: You are expert in coding assistant. Your task is fix the error of this pipeline code.',
                      'Input: The user will provide a pipeline code enclosed in "<CODE> pipline code will be here. </CODE>", '
                      'and an error message enclosed in "<ERROR> error message will be here. </ERROR>".',
                      f"Rule : {StaticValues.rule_code_block}"]

        min_length = min(len(self.pipeline_error), 2000)
        self.small_error_msg = self.pipeline_error[:min_length]
