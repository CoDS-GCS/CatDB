from openai import OpenAI
import os
import tiktoken
from util.Config import LLMSetting


class GenerateLLMCode(object):
    def __init__(self, model: str):
        setting = LLMSetting()
        self.model = model
        self.model_token_limit = setting.get_limit(model=model)

    def generate_llm_code(self, prompt_message: str, prompt_rules: str):
        if self.model == "skip":
            return ""
        elif (self.model == "gpt-3.5-turbo" or
              self.model == "gpt-4" or
              self.model == "gpt-4-turbo"):
            return self.generate_code_OpenAI_LLM(prompt_message=prompt_message, prompt_rules=prompt_rules)
        else:
            raise Exception(f"Model {self.model} is not implemented yet!")

    def generate_code_OpenAI_LLM(self, prompt_message: str, prompt_rules: str):
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), )
        max_token_size = int(self.model_token_limit * 0.95)
        enc = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.encoding_for_model(self.model)
        all_msg = prompt_message + prompt_rules
        token_integers = enc.encode(all_msg)
        num_tokens = len(token_integers)
        messages = [
            {
                "role": "system",
                "content": prompt_rules,
            }
        ]
        if num_tokens >= max_token_size:

            # Split the token integers into chunks based on max_tokens
            chunks = []
            for i in range(0, num_tokens, max_token_size):
                chunks.append(token_integers[i: min(i + max_token_size, num_tokens)])

            # Decode token chunks back to strings
            chunks = [enc.decode(chunk) for chunk in chunks]
            self.submit_Request_OpenAI_LLM(client=client, messages=messages)

            c = 1
            message_text_all = []
            codes = []
            chunk_len = len(chunks)
            for chunk in chunks:
                message_text = []
                if c < chunk_len:
                    message_text.append(
                        f'Do not answer yet. This is just another part of the text I want to send you. Just recive and acknowledge a "Part {c}/{chunk_len} received" and wait for the next part.')

                message_text.append(f'[START PART {c}/{len(chunks)}]')
                message_text.append(f'{chunk}')
                message_text.append(f'[END PART {c}/{len(chunks)}]\n')

                if c < chunk_len:
                    message_text.append(
                        f'Remember not answer yet. Just acknowledge you recived this part with the message "Part {c}/{chunk_len} received" and wait for the next part.')
                if c == chunk_len:
                    message_text.append("ALL PARTS SENT. Now you can continue processing the rquest.")

                message_text = "\n".join(message_text)
                message_text_all.append(message_text)

                messages = [{"role": "user", "content": message_text}]
                # codes.append(send_Message_to_GPT(client=client, messages=messages, model=model))
                c += 1
            code = ""  # TODO: fix large message problem
            print(f" DON'T SEND (Token Size = {num_tokens})!\n")
            return code
        else:
            # prompt_text = f'[START PART 1/1]\n{prompt_text}\n[END PART 1/1]\n[ALL PARTS SENT]'
            messages.append({"role": "user", "content": prompt_message, })
            code = self.submit_Request_OpenAI_LLM(messages=messages, client=client)
            return code

    def submit_Request_OpenAI_LLM(self, messages, client):
        completion = client.chat.completions.create(
                     messages=messages,
                     model=self.model,
                     temperature=0
                    )
        code = completion.choices[0].message.content
        code = code.replace("```", "# ```")
        # code = ""
        return code

    def get_number_tokens(self, prompt_rules: str, prompt_message: str):
        enc = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.encoding_for_model(self.model)
        token_integers = enc.encode(prompt_rules + prompt_message)
        num_tokens = len(token_integers)

        return num_tokens