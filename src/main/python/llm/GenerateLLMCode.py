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
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), )

    if model == "gpt-4":
        model_token_limit = 8192
    elif model == "gpt-3.5-turbo":
        model_token_limit = 4096
    else:
        model_token_limit = -1
    max_token_size = int(model_token_limit * 0.7)
    enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.encoding_for_model(model)
    prompt_format = prompt.format()

    extra_info = prompt.extra_info()
    system_rules = [
        'You will be given a dataset, a schema of the dataset, some extra information, and a question. Your task is to generate a data science pipeline. You should answer only by generating code. You should follow Steps 1 to 11 to answer the question. You should return a data science pipeline in Python 3.10 programming language. If you do not have a relevant answer to the question, simply write: "Insufficient information."',
        'Step 1: the user will supply text, preceded by the prefix "[START PART 1/10]" and followed by the suffix "[END PART 1/10]" (1: refer to first pat, and 10: is the total parts). For instance, \n[START PART 1/10]\nthis is the content of part 1 out of 10.\n[END PART 1/10].'
        'Step 2: when user tell you "[ALL PARTS SENT]", then you can continue processing the data and answering user\'s requests.',
        'Step 3: the user will provide the path of the training and test data enclosed in triple quotes. For the training data, use """train_data=path/to/train/dataset""", and for the test data, use """test_data=path/to/test/dataset""". Load the datasets using pandas\' CSV readers.'
        "Step 4: Don't split the train_data into train and test sets. Use only the given datasets."
        f'Step 5: {prompt_format["rule"]}',
        f'Step 6: {StaticValues.PROMPT_DESCRIPTION.format(extra_info["task_type"], extra_info["task_type"], extra_info["target_attribute"])}',
        f"Step 7: {StaticValues.CODE_FORMATTING_IMPORT}",
        f'Step 8: {StaticValues.CODE_FORMATTING_ADDING.format(extra_info["target_attribute"], extra_info["sample_attribute_names"][0], extra_info["sample_attribute_names"][1])}',
        f"Step 9: {StaticValues.CODE_FORMATTING_DROPPING}",
        f'Step 10: {extra_info["evaluation_text"]}',
        "Step 11: Don't report validation evaluation. We don't need it."]

    prompt_items = [StaticValues.CODE_FORMATTING_DATASET.format(extra_info["data_source_train_path"], extra_info["data_source_test_path"]),
                    "\n",
                    prompt_format["prefix_key"],
                    '"""',
                    prompt_format["prompt"],
                    '"""\n',
                    f'Question: {prompt_format["question"]}']

    prompt_text = "\n".join(prompt_items)
    token_integers = enc.encode(prompt_text)
    num_tokens = len(token_integers)

    system_message = "\n\n".join(system_rules)

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
        for chunk in chunks:
            chunk_text = f'[START PART {c}/{len(chunks)}] \n {chunk} \n[END PART {c}/{len(chunks)}]'
            messages.append([{"role": "user", "content": chunk_text}])
            if c == len(chunks)-1:
                messages.append({"role": "user", "content": "[ALL PARTS SENT]"})

            code = send_Message_to_GPT(client=client, messages=messages, model=model)
            messages.clear()
            c += 1
        return code, system_message, prompt_text
    else:
        prompt_text = f'[START PART 1/1]\n{prompt_text}\n[END PART 1/1]\n[ALL PARTS SENT]'
        messages.append({"role": "user", "content": prompt_text,})
        code = send_Message_to_GPT(messages=messages, client=client, model=model)
        return code, system_message, prompt_text


def send_Message_to_GPT(messages, client, model):
    completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.5
    )
    code = completion.choices[0].message.content
    code = code.replace("```", "# ```")
    return code
