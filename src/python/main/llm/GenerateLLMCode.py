from openai import OpenAI
import os
import tiktoken
from util.Config import LLMSetting
from .GenerateLLMCodeGPT import GenerateLLMCodeGPT
from .GenerateLLMCodeLLaMa import GenerateLLMCodeLLaMa


class GenerateLLMCode(object):
    def __init__(self, model: str):
        setting = LLMSetting()
        self.model = model
        self.model_token_limit = setting.get_limit(model=model)

    def generate_llm_code(self, prompt_message: str, prompt_rules: str):
        if self.model == "skip":
            return ""
        elif (self.model == "gpt-3.5-turbo" or
              self.model == "gpt-4" or
              self.model == "gpt-4-turbo"):
            return GenerateLLMCodeGPT.generate_code_OpenAI_LLM(model=self.model, prompt_message=prompt_message, prompt_rules=prompt_rules)
        elif self.model == "llama3-70b-8192":
            return GenerateLLMCodeLLaMa.generate_code_LLaMa_LLM(model=self.model, prompt_message=prompt_message, prompt_rules=prompt_rules)

        else:
            raise Exception(f"Model {self.model} is not implemented yet!")
