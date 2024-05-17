from openai import OpenAI
import os


class GenerateLLMCodeGPT:
    @staticmethod
    def generate_code_OpenAI_LLM(user_message: str, system_message: str):
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), )
        messages = [
            {"role": "system", "content": system_message} ,
            {"role": "user", "content": user_message}
        ]
        code = GenerateLLMCodeGPT.__submit_Request_OpenAI_LLM( messages=messages, client=client)
        return code

    @staticmethod
    def __submit_Request_OpenAI_LLM(messages, client):
        from util.Config import _llm_model
        completion = client.chat.completions.create(
            messages=messages,
            model=_llm_model,
            temperature=0
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