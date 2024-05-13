class BasicErrorPrompt(object):
    def __init__(self, pipeline_code: str, pipeline_error: str, *args, **kwargs):
        self.rules = []
        self.pipeline_code = pipeline_code
        self.pipeline_error = pipeline_error
        self.small_error_msg = None

    def format_prompt(self):
        return {
            "system_message": self.format_system_message(),
            "user_message": self.format_user_message()
        }

    def format_user_message(self):
        prompt_msg = ["<CODE>\n",
                      self.pipeline_code,
                      "</CODE>\n",
                      "\n",
                      "<ERROR>\n",
                      self.small_error_msg,
                      "</ERROR>\n",
                      "Question: Fix the code error provided and return only the corrected pipeline without "
                      "additional explanations regarding the resolved error.\n"]
        return "".join(prompt_msg)

    def format_system_message(self):
        return "".join(self.rules)


class RuntimeErrorPrompt(BasicErrorPrompt):
    def __init__(self, *args, **kwargs):
        BasicErrorPrompt.__init__(self, *args, **kwargs)
        self.rules = ['You are expert in coding assistant. Your task is fix the error of this pipeline code.\n'
                      'The user will provide a pipeline code enclosed in "<CODE> pipline code will be here. </CODE>", '
                      'and an error message enclosed in "<ERROR> error message will be here. </ERROR>".',
                      'Fix the code error provided and return only the corrected pipeline without additional '
                      'explanations regarding the resolved error.']

        min_length = min(len(self.pipeline_error), 2000)
        self.small_error_msg = self.pipeline_error[:min_length]
