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