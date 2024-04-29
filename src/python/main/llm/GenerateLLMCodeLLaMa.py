from openai import OpenAI
import os
import tiktoken
from util.Config import LLMSetting
import re
from groq import Groq


class GenerateLLMCodeLLaMa:
    @staticmethod
    def generate_code_LLaMa_LLM(model: str, prompt_message: str, prompt_rules: str):
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        client = Groq(api_key=GROQ_API_KEY)
        setting = LLMSetting()
        model_token_limit = setting.get_limit(model=model)

        messages = [
            {"role": "system", "content": prompt_rules},
            {"role": "user", "content": prompt_message}
        ]
        code = GenerateLLMCodeLLaMa.__submit_Request_LLaMa_LLM(model=model, messages=messages, client=client)
        return code

    @staticmethod
    def __submit_Request_LLaMa_LLM(model, messages, client):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        content = completion.choices[0].message.content
        content = GenerateLLMCodeLLaMa.__refine_text(content)
        codes = []
        code_blocks = GenerateLLMCodeLLaMa.__match_code_blocks(content)
        if len(code_blocks) > 0:
            for code in code_blocks:
                codes.append(code)

            return "\n".join(codes)
        else:
            return content

    @staticmethod
    def __match_code_blocks(text):
        pattern = re.compile(r'```(?:python)?[\n\r](.*?)```', re.DOTALL)
        return pattern.findall(text)

    @staticmethod
    def __refine_text(text):
        ind1 = text.find('\n')
        ind2 = text.rfind('\n')

        begin_txt = text[0: ind1]
        end_text = text[ind2+1:len(text)]
        begin_index = 0
        end_index = len(text)
        if begin_txt == "<CODE>":
            begin_index = ind1+1

        if end_text == "</CODE>":
            end_index = ind2
        text = text[begin_index:end_index]
        text = text.replace("<CODE>", "# <CODE>")
        text = text.replace("</CODE>", "# </CODE>")
        return text

    @staticmethod
    def __get_number_tokens(model, prompt_rules: str, prompt_message: str):
        enc = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.encoding_for_model(model)
        token_integers = enc.encode(prompt_rules + prompt_message)
        num_tokens = len(token_integers)
        return num_tokens
