from openai import OpenAI
# client = OpenAI()

client = OpenAI(api_key="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")

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ]
)

print(completion.choices[0].message)
#
# from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain
#
# llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key="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")
#
# template = (
#     "Input:\nQuestion: {question}\nList of Queries:\n{query_list}\nTask:"
# )
#
# prompt = PromptTemplate(
#     input_variables=["question", "query_list"],
#     template=template,
# )
# final_prompt = prompt.format(question="", query_list="")
# chain = LLMChain(llm=llm, prompt=prompt)
#
# question = "Which NFL team won the Super Bowl in the 2010 season?"
#
# query_list_formatted = []
# output = chain.run(question=question, query_list=query_list_formatted)