# import openai
# import os
# from langchain import OpenAI, PromptTemplate, LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationSummaryBufferMemory
# from langchain.chains import ConversationChain
#
# llm = OpenAI(temperature=0)
# conversation_with_summary = ConversationChain(
#     llm=llm,
#     # We set a very low max_token_limit for the purposes of testing.
#     memory=ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40),
#     verbose=True
# )
#
# conversation_with_summary.predict(input='FIRST PROMPT')
# conversation_with_summary.predict(input='SECOND PROMPT')
# ...
# conversation_with_summary.predict(input='N PROMPT')
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

os.environ.get("OPENAI_API_KEY")

llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),model_name="gpt-3.5-turbo", temperature=0)

template = (
    "Input:\nQuestion: {question}\nList of Queries:\n{query_list}\nTask:"
)

prompt = PromptTemplate(
    input_variables=["question", "query_list"],
    template=template,
)
question="aaaaaaaaaaaaaaaaaaaaaaaa"
query_list_formatted=["A", "B"]
final_prompt = prompt.format(question="", query_list="")
chain = LLMChain(llm=llm, prompt=prompt)
output = chain.run(question=question, query_list=query_list_formatted)