from openai import OpenAI


def generate_llm_code(model: str, prompt: str):
    if model == "skip":
        return ""
    elif model == "gpt-3.5-turbo" or model == "gpt-4":
        return generate_GPT_LLM_code(model=model, prompt=prompt)
    else:
        raise Exception(f"Model {model} is not implemented yet!")


def generate_GPT_LLM_code(model: str, prompt: str):
    client = OpenAI(api_key="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", )  # os.environ.get("OPENAI_API_KEY")
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert datascientist assistant generate a pipeline. You answer only by generating code. Answer as concisely as possible.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=model,
    )
    code = completion.choices[0].message.content

    # Refine code
    # llm_extra_texts = ['```python-import',
    #                    '```end-import',
    #                    '```python-load-dataset',
    #                    '```end-load-dataset'
    #                    '```python-added-column',
    #                    ' ```end-added-column',
    #                    '```python-dropping-columns',
    #                    '```end-dropping-columns'
    #                    '```python-training-technique',
    #                    '```end-training-technique',
    #                    '```python-other',
    #                    '```end-other',
    #                    '<end>']
    # for et in llm_extra_texts:
    #     code = code.replace(et, "")
    code = code.replace("```", "# ")
    return code
