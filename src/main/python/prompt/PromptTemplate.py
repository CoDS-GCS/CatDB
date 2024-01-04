from util import StaticValues
import re


class BasicPrompt(object):
    def __init__(self, *args, **kwargs):
        # used to avoid empty init function in 0-shot prompt
        pass

    def format_target(self, example: dict):
        return self.format_question(example)

    def format_question(self, examples: dict):
        raise NotImplementedError()

    def get_extra_info(self, examples: dict):
        return None


class TextPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.template_info = "Suppose there is a dataset with training data \"{}\" and test data \"{}\" on the disk, with columns appropriately named as attributes:\n\n{}"
        self.template_question = "Generate as many features as useful for downstream classifier, \
        but as few as necessary to reach good performance. and can drop unused columns (Feature selection).\
        \n Each codeblock ends with \"```end-*\" and starts with \"```python-*\" \n Return a full pipeline code."

    def format_question(self, examples: dict):
        schema = "\n".join([f"{_} (data type:{self.schema[_]})" for _ in self.schema.keys()])
        schema_keys = [_ for _ in self.schema.keys()]

        prompt_info = self.template_info.format(self.data_source_train_path, self.data_source_test_path, schema)
        prompt_description = StaticValues.PROMPT_DESCRIPTION.format(self.task_type, self.target_attribute)

        prompt_components = [prompt_info,
                             prompt_description,
                             StaticValues.CODE_FORMATTING_IMPORT,
                             StaticValues.CODE_FORMATTING_REQUIREMENTS,
                             StaticValues.CODE_FORMATTING_ADDING.format(self.target_attribute,
                                                                        schema_keys[0],
                                                                        schema_keys[1]),
                             StaticValues.CODE_FORMATTING_DROPPING,
                             StaticValues.CODE_FORMATTING_TECHNIQUE.format(self.task_type),
                             StaticValues.CODE_FORMATTING_OTHER,
                             StaticValues.CODE_FORMATTING_LOAD_DATASET.format(self.file_format),
                             self.template_question]
        prompt = "\n\n".join(prompt_components)
        return re.sub(' +', ' ', prompt)
