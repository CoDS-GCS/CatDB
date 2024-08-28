from .PreprocessingLLMGemini import DataPrepareLLMGemini


class DataPrepareLLM:

    @staticmethod
    def data_prepare_llm(user_message: str, system_message: str):
        from util.Config import _llm_platform, _OPENAI, _META, _GOOGLE

        if _llm_platform is None:
            raise Exception("Select a LLM Platform: OpenAI (GPT) or Meta (Lama)")
        # elif _llm_platform == _OPENAI:
        #     return GenerateLLMCodeGPT.generate_code_OpenAI_LLM(user_message=user_message, system_message=system_message)
        # elif _llm_platform == _META:
        #     return GenerateLLMCodeLLaMa.generate_code_LLaMa_LLM(user_message=user_message, system_message=system_message)
        elif _llm_platform == _GOOGLE:
            return DataPrepareLLMGemini.DataPrepare_Gemini_LLM(user_message=user_message, system_message=system_message)

        else:
            raise Exception(f"Model {_llm_platform} is not implemented yet!")

    # @staticmethod
    # def refine_source_code(code: str):
    #     final_code = []
    #     for line in code.splitlines():
    #         if not line.startswith('#'):
    #             final_code.append(line)
    #     final_code = "\n".join(final_code)
    #     return final_code.replace("@ ```", "# ```")

    # def get_number_tokens(user_message: str, system_message: str):
    #     from util.Config import _llm_platform, _OPENAI, _META, _GOOGLE
    #
    #     if _llm_platform is None:
    #         raise Exception("Select a LLM Platform: OpenAI (GPT) or Meta (Lama)")
    #     elif _llm_platform == _OPENAI:
    #         return GenerateLLMCodeGPT.get_number_tokens(user_message=user_message, system_message=system_message)
    #     elif _llm_platform == _META:
    #         return GenerateLLMCodeLLaMa.get_number_tokens(user_message=user_message, system_message=system_message)
    #     elif _llm_platform == _GOOGLE:
    #         prompt = [system_message, user_message]
    #         message = "\n".join(prompt)
    #         return GenerateLLMGemini.get_number_tokens(messages=message)
    #
    #     else:
    #         raise Exception(f"Model {_llm_platform} is not implemented yet!")
