from .DataPrepareLLMGemini import DataPrepareLLMGemini


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

    @staticmethod
    def extract_row_col_values(result: str):
        rows = result.splitlines()
        values = dict()
        for row in rows:
            rds = row.split(":")
            row_index = int(rds[0].replace("### Row ", ""))
            cds = rds[1].split(",")
            col_values = dict()
            for col in cds:
                cs = col.split("=")
                cname = cs[0].replace(" ", "")
                cvalue = cs[1]
                col_values[cname] = cvalue
            values[row_index] = col_values

        return values

    @staticmethod
    def extract_catalog_values(result: str):
        rows = result.splitlines()
        values = dict()
        for row in rows:
            if row == '```':
                continue
            try:
                rds = row.split(":")
                col_name = rds[0]
                result = 'none-categorical'
                if rds[1].strip().lower() == "yes":
                    result = 'categorical'
                elif rds[1].strip().lower() == "list":
                    result = 'list'
                values[col_name] = result
            except:
                pass
        return values



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
