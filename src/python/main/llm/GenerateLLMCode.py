from .GenerateLLMCodeGPT import GenerateLLMCodeGPT
from .GenerateLLMCodeLLaMa import GenerateLLMCodeLLaMa
import tiktoken


class GenerateLLMCode:

    @staticmethod
    def generate_llm_code(user_message: str, system_message: str):
        from util.Config import _llm_platform, _OPENAI, _META

        if _llm_platform is None:
            raise Exception("Select a LLM Platform: OpenAI (GPT) or Meta (Lama)")
        elif _llm_platform == _OPENAI:
            return GenerateLLMCodeGPT.generate_code_OpenAI_LLM(user_message=user_message, system_message=system_message)
        elif _llm_platform == _META:
            return GenerateLLMCodeLLaMa.generate_code_LLaMa_LLM(user_message=user_message, system_message=system_message)

        else:
            raise Exception(f"Model {_llm_platform} is not implemented yet!")

    @staticmethod
    def __get_number_tokens(user_message: str, system_message: str):
        from util.Config import _llm_platform
        enc = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.encoding_for_model(_llm_platform)
        token_integers = enc.encode(user_message + system_message)
        num_tokens = len(token_integers)
        return num_tokens
