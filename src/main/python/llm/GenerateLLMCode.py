from openai import OpenAI
import os
import tiktoken


def generate_llm_code(model: str, prompt: str):
    if model == "skip":
        return ""
    elif model == "gpt-3.5-turbo" or model == "gpt-4":
        return generate_GPT_LLM_code(model=model, prompt=prompt)
    else:
        raise Exception(f"Model {model} is not implemented yet!")


def generate_GPT_LLM_code(model: str, prompt: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), )

    if model == "gpt-4":
        model_token_limit = 8192
    elif model == "gpt-3.5-turbo":
        model_token_limit = 4096
    else:
        model_token_limit = -1
    max_token_size = int(model_token_limit * 0.9)
    enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.encoding_for_model(model)
    token_integers = enc.encode(prompt)
    num_tokens = len(token_integers)

    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant generate a pipeline. You answer only by generating code. Answer as concisely as possible.",
        }
    ]
    if num_tokens >= model_token_limit:

        # Split the token integers into chunks based on max_tokens
        chunks = []
        for i in range(0, num_tokens, max_token_size):
            chunks.append(token_integers[i: min(i + max_token_size, num_tokens)])

        # Decode token chunks back to strings
        chunks = [enc.decode(chunk) for chunk in chunks]
        instruction_txt_list = ["The total length of the content that I want to send you is too large to send in only one piece.",
                                "sending you that content, I will follow this rule:",
                                f"[START PART 1/{len(chunks)}]",
                                f"this is the content of the part 1 out of {len(chunks)} in total",
                                f"[END PART 1/{len(chunks)}]",
                                "And when I tell you \"ALL PARTS SENT\", then you can continue processing the data and answering my requests."]

        instruction_txt = "\n".join(instruction_txt_list)
        messages.append({"role": "system", "content": instruction_txt})
        send_Message_to_GPT(client=client, messages=messages, model=model)

        c = 1
        code = ""
        for chunk in chunks:
            chunk_text = f"[START PART {c}/{len(chunks)}] \n {chunk} \n[END PART {c}/{len(chunks)}]"
            messages=[{"role": "user", "content": chunk_text}]
            # if c == len(chunks):
            #     messages.append({"role": "user", "content": "ALL PARTS SENT"})

            code = send_Message_to_GPT(client=client, messages=messages, model=model)
            c += 1
        return code
    else:
        messages.append({"role": "user", "content": prompt,})
        code = send_Message_to_GPT(messages=messages, client=client, model=model)
        return code


def send_Message_to_GPT(messages, client, model):
    completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.5
    )
    code = completion.choices[0].message.content
    code = code.replace("```", "# ")
    return code
