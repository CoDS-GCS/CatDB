import os

# from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain

from openai import OpenAI


if __name__ == '__main__':
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="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",
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )

    # api_key = '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'
    # openai.api_key = api_key
    #
    # # Define your prompt
    # prompt = "Translate the following English text to French:"
    #
    # # Define the model and other parameters
    # model = "text-davinci-003"  # You can choose a different model
    # temperature = 0.7
    # max_tokens = 100
    #
    # # Submit the prompt to OpenAI API
    # response = openai.Completion.create(
    #     engine=model,
    #     prompt=prompt,
    #     temperature=temperature,
    #     max_tokens=max_tokens
    # )
    #
    # # Get the generated text from the response
    # generated_text = response['choices'][0]['text']
    #
    # # Print the generated text
    # print("Generated Text:")
    # print(generated_text)
    #
    # client = OpenAI(api_key="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")

    # client.api_key
    # os.environ["OPENAI_API_KEY"] = "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"
    #
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[],
    #     temperature=0,
    #     max_tokens=1024
    # )

    #
    #
    # llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
    #
    # template = (
    #     "Input:\nQuestion: {question}\nList of Queries:\n{query_list}\nTask:"
    # )
    #
    # prompt = PromptTemplate(
    #     input_variables=["question", "query_list"],
    #     template=template,
    # )
    # final_prompt = prompt.format(
    #     question="Write a Python function that takes as input a file path to an image, loads the image into memory as a numpy array, then crops the rows and columns around the perimeter if they are darker than a threshold value. Use the mean value of rows and columns to decide if they should be marked for deletion.",
    #     query_list="")
    # chain = LLMChain(llm=llm, prompt=prompt)
    # # output = chain.run(question=question, query_list=query_list_formatted)
