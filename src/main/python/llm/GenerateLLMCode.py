
def generate_llm_code(messages):
    if model == "skip":
        return ""

    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stop=["```end"],
        temperature=0.5,
        max_tokens=500,
    )
    code = completion["choices"][0]["message"]["content"]
    code = code.replace("```python", "").replace("```", "").replace("<end>", "")
    return code