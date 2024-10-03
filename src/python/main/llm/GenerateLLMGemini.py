import google.generativeai as genai
import os
import time


class GenerateLLMGemini:
    @staticmethod
    def generate_code_Gemini_LLM(user_message: str, system_message: str):
        from util.Config import _LLM_API_Key
        _, api_key = _LLM_API_Key.get_API_Key()
        genai.configure(api_key=api_key)
        code, number_of_tokens, time_gen = GenerateLLMGemini.__submit_Request_Gemini_LLM(user_message=user_message, system_message=system_message)
        return code, number_of_tokens, time_gen

    @staticmethod
    def __submit_Request_Gemini_LLM(user_message: str, system_message: str):
        from util.Config import _LLM_API_Key, _llm_model, _temperature, _top_p, _top_k, _max_out_token_limit

        time_start = time.time()

        generation_config = {
            "temperature": _temperature,
            "top_p": _top_p,
            "top_k": _top_k,
            "max_output_tokens": _max_out_token_limit,
        }

        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        # ]
        model = genai.GenerativeModel(model_name=_llm_model,
                                      generation_config=generation_config,
                                      #safety_settings=safety_settings,
                                      system_instruction=system_message)
        try:
            prompt = [system_message, user_message]
            message = "\n".join(prompt)
            number_of_tokens = model.count_tokens(message).total_tokens

            chat_session = model.start_chat(
                history=[{
                    "role": "user",
                    "parts": [user_message],
                }]
            )

            response = chat_session.send_message("INSERT_INPUT_HERE")
            code = response.text
            # Refine code, keep all codes are between ```python and ```end
            begin_key = "```python"
            end_key = "```end"[::-1]
            begin_point = code.find(begin_key)
            end_point = len(code) - code[::-1].find(end_key)
            code = code[begin_point:end_point]
            code = code.replace("```", "@ ```")

            from .GenerateLLMCode import GenerateLLMCode
            code = GenerateLLMCode.refine_source_code(code=code)
            time_end = time.time()
            return code, number_of_tokens, time_end - time_start

        except Exception as err:
            _, api_key = _LLM_API_Key.get_API_Key()
            print("---------------------------")
            print(err)
            print(api_key)
            print("******************************************")
            genai.configure(api_key=api_key)
            return GenerateLLMGemini.__submit_Request_Gemini_LLM(user_message=user_message, system_message=system_message)

    @staticmethod
    def get_number_tokens(messages: str):
        from util.Config import _llm_model, _temperature, _top_p, _top_k, _max_out_token_limit

        generation_config = {
            "temperature": _temperature,
            "top_p": _top_p,
            "top_k": _top_k,
            "max_output_tokens": _max_out_token_limit,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        model = genai.GenerativeModel(model_name=_llm_model,
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        number_of_tokens = model.count_tokens(messages).total_tokens
        return number_of_tokens
