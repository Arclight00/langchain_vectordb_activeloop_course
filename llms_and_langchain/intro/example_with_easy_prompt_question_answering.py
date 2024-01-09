# Examples with Easy Prompts: Text Summarization, Text Translation, and Question Answering

"""
In the realm of natural language processing, Large Language Models have become a popular tool for tackling
various text-based tasks. These models can be promoted in different ways to produce a range of results,
depending on the desired outcome.

Setting Up the Environment

To begin, we will need to install the huggingface_hubCopy library in addition to previously installed packages
and dependencies. Also, keep in mind to create the Huggingface API Key by navigating to Access Tokens page under
the account’s Settings. The key must be set as an environment variable with HUGGINGFACEHUB_API_TOKEN key.
pip install -q huggingface_hub
"""
from config import *

# Creating a Question-Answering Prompt Template

from langchain import PromptTemplate
from langchain.llms import OpenAI

template = """
Question: {question}
Answer: 
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "What is the capital city of France?"

"""
Next, we will use the Hugging Face model google/flan-t5-large to answer the question. 
The HuggingfaceHub class will connect to Hugging Face’s inference API and load the specified model.
"""
from langchain import HuggingFaceHub, LLMChain

# initialize Hub LLM
hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0}
)
# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about the capital of France
# print(llm_chain.run(question))

# We can also modify the prompt template to include multiple questions.

"""
Asking Multiple Questions

To ask multiple questions, we can either iterate through all questions one at a time or place all 
questions into a single prompt for more advanced LLMs. 

Let's start with the first approach:
"""
qa = [
    {'question': "What is the capital city of France?"},
    {'question': "What is the largest mammal on Earth?"},
    {'question': "Which gas is most abundant in Earth's atmosphere?"},
    {'question': "What color is a ripe banana?"}
]

# res = llm_chain.generate(qa)
# print(res)

"""
We can modify our prompt template to include multiple questions to implement a second approach. 
The language model will understand that we have multiple questions and answer them sequentially. 
This method performs best on more capable models.
"""

multi_template = """Answer the following questions one at a time
Questions: 
{questions}

Answers:
"""

long_prompt = PromptTemplate(template=multi_template, input_variables=['questions'])
llm = OpenAI(model_name="text-davinci-003", temperature=0)

llm_chain2 = LLMChain(
    prompt=long_prompt,
    llm=llm
)

qs_str = (
 "What is the capital city of France?\n" +
 "What is the largest mammal on Earth?\n" +
 "Which gas is most abundant in Earth's atmosphere?\n" +
 "What color is a ripe banana?\n"
)

print(llm_chain2.run(qs_str))