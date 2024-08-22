import google.generativeai as genai
import os
import time
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

class FineTuneLLMGemini:

    @staticmethod
    def load_creds():
        from util.Config import Google_SCOPES, _google_client_secret_file_path, _google_token_file_path
        """Converts `client_secret.json` to a credential object.

        This function caches the generated tokens to minimize the use of the
        consent screen.
        """
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(_google_token_file_path):
            creds = Credentials.from_authorized_user_file(_google_token_file_path, Google_SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(_google_client_secret_file_path, Google_SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(_google_token_file_path, 'w') as token:
                token.write(creds.to_json())
        return creds

    @staticmethod
    def finetune_Gemini_LLM(train_data: None, model_name: None):
        import random
        from util.Config import _llm_model
        creds = FineTuneLLMGemini.load_creds()
        genai.configure(credentials=creds)
        name = f'generate-num-{random.randint(0, 10000)}'
        # base_model = [
        #     m for m in genai.list_models()
        #     if "createTunedModel" in m.supported_generation_methods][0]

        # remove last model
        try:
            genai.delete_tuned_model(f'tunedModels/{model_name}')
        except Exception as e:
            pass

        operation = genai.create_tuned_model(
            source_model=f'models/{_llm_model}',
            training_data=train_data,
            id=model_name,
            epoch_count=100,
            batch_size=4,
            learning_rate=0.001,
        )



        # model = genai.get_tuned_model(f'tunedModels/{model_name}dddd')
        # print(model)
        # from util.Config import _LLM_API_Key
        # _, api_key = _LLM_API_Key.get_API_Key()
        # genai.configure(api_key=api_key)
        # prompt = [system_message, user_message]
        # message = "\n".join(prompt)
        # code, number_of_tokens, time_gen = GenerateLLMGemini.__submit_Request_Gemini_LLM(messages=message)
        # return code, number_of_tokens, time_gen
    #
    # @staticmethod
    # def __submit_Request_Gemini_LLM(messages):
    #     from util.Config import _LLM_API_Key, _llm_model, _temperature, _top_p, _top_k, _max_out_token_limit
    #
    #     time_start = time.time()
    #
    #     generation_config = {
    #         "temperature": _temperature,
    #         "top_p": _top_p,
    #         "top_k": _top_k,
    #         "max_output_tokens": _max_out_token_limit,
    #     }
    #
    #     safety_settings = [
    #         {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #         {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #         {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #     ]
    #
    #     model = genai.GenerativeModel(model_name=_llm_model,
    #                                   generation_config=generation_config,
    #                                   safety_settings=safety_settings)
    #
    #     number_of_tokens = model.count_tokens(messages).total_tokens
    #
    #     try:
    #         response = model.generate_content(messages)
    #         code = response.text
    #         # Refine code, keep all codes are between ```python and ```end
    #         begin_key = "```python"
    #         end_key = "```end"[::-1]
    #         begin_point = code.find(begin_key)
    #         end_point = len(code) - code[::-1].find(end_key)
    #         code = code[begin_point:end_point]
    #         code = code.replace("```", "@ ```")
    #
    #         from .GenerateLLMCode import GenerateLLMCode
    #         code = GenerateLLMCode.refine_source_code(code=code)
    #         time_end = time.time()
    #         return code, number_of_tokens, time_end - time_start
    #
    #     except Exception:
    #         _, api_key = _LLM_API_Key.get_API_Key()
    #         genai.configure(api_key=api_key)
    #         return GenerateLLMGemini.__submit_Request_Gemini_LLM(messages)
    #
    # @staticmethod
    # def get_number_tokens(messages: str):
    #     from util.Config import _llm_model, _temperature, _top_p, _top_k, _max_out_token_limit
    #
    #     generation_config = {
    #         "temperature": _temperature,
    #         "top_p": _top_p,
    #         "top_k": _top_k,
    #         "max_output_tokens": _max_out_token_limit,
    #     }
    #
    #     safety_settings = [
    #         {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #         {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #         {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #     ]
    #
    #     model = genai.GenerativeModel(model_name=_llm_model,
    #                                   generation_config=generation_config,
    #                                   safety_settings=safety_settings)
    #
    #     number_of_tokens = model.count_tokens(messages).total_tokens
    #     return number_of_tokens
