from config import *

"""
LangChain prompts can be found in various use cases, such as summarization or question-answering chains. 
For example, when creating a summarization chain, LangChain enables interaction with an external data source 
to fetch data for use in the generation step. This could involve summarizing a lengthy piece of text or answering 
questions using specific data sources.

The following code will initialize the language model using OpenAI class with a temperature of 0 - because we want 
deterministic output.  The load_summarize_chain function accepts an instance of the language model and returns a 
pre-built summarization chain. Lastly, the PyPDFLoader class is responsible for loading PDF files and converting 
them into a format suitable for processing by LangChain. 
"""

from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

# Initialize language model
llm = OpenAI(model_name='text-davinci-003', temperature=0)

# Load the summarization chain
summarize_chain = load_summarize_chain(llm)

# Load the document using PyPDFLoader
document_loader = PyPDFLoader(file_path="/Users/nishantsingh/Desktop/Nishant Documents/Nishant Singh.pdf")
document = document_loader.load()

# Summarize the document
summary = summarize_chain(document)
print(summary['output_text'])

"""
In this example, the code uses the default summarization chain provided by the load_summarize_chain function. 
However, you can customize the summarization process by providing prompt templates.
"""
