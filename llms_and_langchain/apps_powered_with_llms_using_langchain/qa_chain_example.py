from config import *

"""
We can also use LangChain to manage prompts for asking general questions from the LLMs. These models are proficient 
in addressing fundamental inquiries. Nevertheless, it is crucial to remain mindful of the potential issue of 
hallucinations, where the models may generate non-factual information. To address this concern, we will later 
introduce the Retrieval chain as a means to overcome this problem.
"""

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

prompt_template = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=['question'])

llm = OpenAI(model_name='text-davinci-003', temperature=0)

chain = LLMChain(prompt=prompt_template, llm=llm)

"""
We define a custom prompt template by creating an instance of the PromptTemplate class. The template string contains 
a placeholder {question} for the input question, followed by a newline character and the "Answer:" label.  
The input_variables argument is set to the list of available placeholders in the prompt (like a question in this case) 
to indicate the name of the variable that the chain will replace in the template.run() method.

We then instantiate an OpenAI model named text-davinci-003 with a temperature of 0. The OpenAI class is used to 
create the instance, and the model_name and temperature arguments are provided. Finally, we create a question-answering
chain using the LLMChain class. 

The class constructor takes two arguments: llm, which is the instantiated OpenAI model, and prompt, which is the 
custom prompt template we defined earlier. 

By following these steps, we can process input questions effectively with the custom question-answering, generating 
appropriate answers using the OpenAI model and the custom prompt template.
"""

print(chain.run("what is the meaning of life?"))