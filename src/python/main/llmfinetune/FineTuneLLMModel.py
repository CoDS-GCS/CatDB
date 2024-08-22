from .FineTuneLLMGemini import FineTuneLLMGemini


class FineTuneLLMModel:

    @staticmethod
    def finetune_llm_model(model_name: None):
        from util.Config import _llm_platform, _OPENAI, _META, _GOOGLE

        if _llm_platform is None:
            raise Exception("Select a LLM Platform: OpenAI (GPT) or Meta (Lama)")
        elif _llm_platform == _OPENAI:
            pass
        elif _llm_platform == _META:
            pass
        elif _llm_platform == _GOOGLE:

            return FineTuneLLMGemini.finetune_Gemini_LLM(train_data=FineTuneLLMModel.convert_df_to_train_data(),
                                                         model_name=model_name)

        else:
            raise Exception(f"Model {_llm_platform} is not implemented yet!")

    @staticmethod
    def convert_df_to_train_data():
        from util.Config import _fintune_train_data, _fintune_train_data_target_attribute
        train_data = []
        columns = list(_fintune_train_data.columns.values)

        for index, row in _fintune_train_data.iterrows():
            output_text = f'{row[_fintune_train_data_target_attribute]}'
            input_texts = []
            for column in columns:
                if column in _fintune_train_data_target_attribute:
                    continue
                input_texts.append(f"{column} is {row[column]}")

            train_data.append({'text_input': ",".join(input_texts), 'output': output_text})

        return train_data
