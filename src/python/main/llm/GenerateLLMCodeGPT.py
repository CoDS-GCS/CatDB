from openai import OpenAI
import tiktoken
import time


class GenerateLLMCodeGPT:
    @staticmethod
    def generate_code_OpenAI_LLM(user_message: str, system_message: str):
        from util.Config import _LLM_API_Key
        _, api_key = _LLM_API_Key.get_API_Key()
        client = OpenAI(api_key=api_key )
        number_of_tokens = GenerateLLMCodeGPT.get_number_tokens(user_message=user_message, system_message=system_message)
        messages = [
            {"role": "system", "content": system_message} ,
            {"role": "user", "content": user_message}
        ]
        time_start = time.time()
        code = GenerateLLMCodeGPT.__submit_Request_OpenAI_LLM( messages=messages, client=client)
        time_end = time.time()
        return code, number_of_tokens, time_end - time_start

    @staticmethod
    def __submit_Request_OpenAI_LLM(messages, client):
        from util.Config import _llm_model, _temperature
        completion = client.chat.completions.create(
            messages=messages,
            model=_llm_model,
            temperature = _temperature
        )
        code = completion.choices[0].message.content
        # Refine code, keep all codes are between ```python and ```end
        begin_key = "```python"
        end_key = "```end"[::-1]
        begin_point = code.find(begin_key)
        end_point = len(code) - code[::-1].find(end_key)
        code = code[begin_point:end_point]
        code = code.replace("```", "@ ```")

        from .GenerateLLMCode import GenerateLLMCode
        code = GenerateLLMCode.refine_source_code(code=code)
        return code

    @staticmethod
    def get_number_tokens(user_message: str, system_message: str):
        from util.Config import _llm_model
        enc = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.encoding_for_model(_llm_model)
        token_integers = enc.encode(user_message + system_message)
        num_tokens = len(token_integers)
        return num_tokens