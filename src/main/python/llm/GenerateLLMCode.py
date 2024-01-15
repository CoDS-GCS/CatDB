from openai import OpenAI
import os
import tiktoken
from util import StaticValues
from prompt import BasicICLPrompt


def generate_llm_code(model: str, prompt: BasicICLPrompt):
    if model == "skip":
        return ""
    elif model == "gpt-3.5-turbo" or model == "gpt-4":
        return generate_GPT_LLM_code(model=model, prompt=prompt)
    else:
        raise Exception(f"Model {model} is not implemented yet!")


def generate_GPT_LLM_code(model: str, prompt: BasicICLPrompt):

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)

    if model == "gpt-4":
        model_token_limit = 8192
    elif model == "gpt-3.5-turbo":
        model_token_limit = 4096
    else:
        model_token_limit = -1


    max_token_size = model_token_limit * 0.8
    enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.encoding_for_model(model)
    prompt_format = prompt.format()

    extra_info = prompt.extra_info()
    prompt_items = [StaticValues.CODE_FORMATTING_DATASET.format(extra_info["data_source_train_path"],
                                                                extra_info["data_source_test_path"]),
                    "\n",
                    prompt_format["prefix_key"],
                    '"""',
                    prompt_format["prompt"],
                    '"""\n',
                    f'Question: {prompt_format["question"]}']

    prompt_text = "\n".join(prompt_items)
    token_integers = enc.encode(prompt_text)
    num_tokens = len(token_integers)

    system_rules = ['You will be given a dataset, a schema of the dataset, some extra information, and a question. Your task is to generate a data science pipeline. You should answer only by generating code. You should follow Steps 1 to 11 to answer the question. You should return a data science pipeline in Python 3.10 programming language. If you do not have a relevant answer to the question, simply write: "Insufficient information."']
    if num_tokens >= model_token_limit:
        system_rules.append('the user will supply text, preceded by the prefix "[START PART 1/10]" and followed by the suffix "[END PART 1/10]" (1: refer to first pat, and 10: is the total parts). For instance, \n[START PART 1/10]\nthis is the content of part 1 out of 10.\n[END PART 1/10].')
        system_rules.append('when user tell you "ALL PARTS SENT", then you can continue processing the data and answering user\'s requests.')

    system_rules.extend([
        'the user will provide the path of the training and test data enclosed in triple quotes. For the training data, use """train_data=path/to/train/dataset""", and for the test data, use """test_data=path/to/test/dataset""". Load the datasets using pandas\' CSV readers.',
        "Don't split the train_data into train and test sets. Use only the given datasets.",
        f'{prompt_format["rule"]}',
        f'{StaticValues.PROMPT_DESCRIPTION.format(extra_info["task_type"], extra_info["task_type"], extra_info["target_attribute"])}',
        f"{StaticValues.CODE_FORMATTING_IMPORT}",
        f'{StaticValues.CODE_FORMATTING_ADDING.format(extra_info["target_attribute"], extra_info["sample_attribute_names"][0], extra_info["sample_attribute_names"][1])}',
        f"{StaticValues.CODE_FORMATTING_DROPPING}",
        f'{extra_info["evaluation_text"]}',
        "Don't report validation evaluation. We don't need it."])

    system_message = ""
    step_index = 1
    for m in system_rules:
        system_message += f"Step {step_index}: {m} \n\n"
        step_index +=1

    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]
    if num_tokens >= model_token_limit:

        # Split the token integers into chunks based on max_tokens
        chunks = []
        for i in range(0, num_tokens, max_token_size):
            chunks.append(token_integers[i: min(i + max_token_size, num_tokens)])

        # Decode token chunks back to strings
        chunks = [enc.decode(chunk) for chunk in chunks]
        send_Message_to_GPT(client=client, messages=messages, model=model)

        c = 1
        code = ""
        message_text_all = []
        codes = []
        chunk_len = len(chunks)
        for chunk in chunks:
            message_text = []
            if c < chunk_len:
                message_text.append(f'Do not answer yet. This is just another part of the text I want to send you. Just recive and acknowledge a "Part {c}/{chunk_len} received" and wait for the next part.')

            message_text.append(f'[START PART {c}/{len(chunks)}]')
            message_text.append(f'{chunk}')
            message_text.append(f'[END PART {c}/{len(chunks)}]\n')

            if c < chunk_len:
                message_text.append(f'Remember not answer yet. Just acknowledge you recived this part with the message "Part {c}/{chunk_len} received" and wait for the next part.')
            if c == chunk_len:
                message_text.append("ALL PARTS SENT. Now you can continue processing the rquest.")

            message_text = "\n".join(message_text)
            message_text_all.append(message_text)

            messages = [{"role": "user", "content": message_text}]
            codes.append(send_Message_to_GPT(client=client, messages=messages, model=model))
            codes.append("#===============================================================")
            c += 1
        return code, system_message, "\n".join(message_text_all)
    else:
        #prompt_text = f'[START PART 1/1]\n{prompt_text}\n[END PART 1/1]\n[ALL PARTS SENT]'
        messages.append({"role": "user", "content": prompt_text,})
        code = send_Message_to_GPT(messages=messages, client=client, model=model)
        return code, system_message, prompt_text


def send_Message_to_GPT(messages, client, model):
    completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.2,
        top_p=0.1,

    )
    code = completion.choices[0].message.content
    code = code.replace("```", "# ```")
    return code
