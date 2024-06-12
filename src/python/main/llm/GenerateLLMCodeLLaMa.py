import os
import re
from groq import Groq
import time


class GenerateLLMCodeLLaMa:
    @staticmethod
    def generate_code_LLaMa_LLM(user_message: str, system_message: str):
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        client = Groq(api_key=GROQ_API_KEY)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        time_start = time.time()
        code = GenerateLLMCodeLLaMa.__submit_Request_LLaMa_LLM(messages=messages, client=client)
        time_end = time.time()
        return code, 0, time_end - time_start

    @staticmethod
    def __submit_Request_LLaMa_LLM(messages, client):
        from util.Config import _llm_model, _delay
        try:
            completion = client.chat.completions.create(
                model=_llm_model,
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
        except Exception as err:
            time.sleep(_delay)
            return GenerateLLMCodeLLaMa.__submit_Request_LLaMa_LLM(messages, client)

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
        text = text.replace("```", "@ ```")

        from .GenerateLLMCode import GenerateLLMCode
        text = GenerateLLMCode.refine_source_code(code=text)
        return text