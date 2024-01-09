"""
Using LangChain, we can create a chain for text summarization.

First, we need to set up the necessary imports and an instance of the OpenAI language model:
"""
from config import *
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# Next, we define a prompt template for summarization:
summarization_template = "Summarize the following text to one sentence: {text}"
summarization_prompt = PromptTemplate(template=summarization_template, input_variables=['text'])
summarization_chain = LLMChain(prompt=summarization_prompt, llm=llm)

# To use the summarization chain, simply call the predict method with the text to be summarized:

text = "LangChain provides many modules that can be used to build language model applications. " \
       "Modules can be combined to create more complex applications, or be used individually for simple applications. " \
       "The most basic building block of LangChain is calling an LLM on some input. " \
       "Let’s walk through a simple example of how to do this. For this purpose, " \
       "let’s pretend we are building a service that generates a company name based on what the company makes."

summarized_text = summarization_chain.predict(text=text)
print(summarized_text)