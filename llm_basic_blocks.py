from langchain.llms import OpenAI
from config import active_loop_key, openai_api_key


'''
The temperature parameter in OpenAI models manages the randomness of the output. 
When set to 0, the output is mostly predetermined and suitable for tasks requiring stability and the most probable result. 
At a setting of 1.0, the output can be inconsistent and interesting but isn't generally advised for most tasks. 
For creative tasks, a temperature between 0.70 and 0.90 offers a balance of reliability and creativity. 
The best setting should be determined by experimenting with different values for each specific use case.
'''
llm = OpenAI(model='text-davinci-003', temperature=0.9)
text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
# print(llm(text))

## The Chain
'''
In LangChain, a chain is an end-to-end wrapper around multiple individual components,
providing a way to accomplish a common use case by combining these components in a specific sequence. 
The most commonly used type of chain is the LLMChain, which consists of a PromptTemplate, 
a model (either an LLM or a ChatModel), and an optional output parser.

The LLMChain works as follows:

--->Takes (multiple) input variables.
--->Uses the PromptTemplate to format the input variables into a prompt.
--->Passes the formatted prompt to the model (LLM or ChatModel).
--->If an output parser is provided, it uses the OutputParser to parse the output of the LLM into a final format.
--->In the next example, we demonstrate how to create a chain that generates a possible name for a company that 
produces eco-friendly water bottles. By using LangChain's LLMChain, PromptTemplate, and OpenAIclasses, 
we can easily define our prompt, set the input variables, and generate creative outputs. 
'''
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=['product'],
    template='What is a good name for a company that makes {product}?'
)

# chain = LLMChain(llm=llm, prompt=prompt)
# Run the chain only specifying the input variable.
# print(chain.run("eco-friendly water bottles"))


## The Memory
'''
In LangChain, Memory refers to the mechanism that stores and manages the conversation history between a user and the AI.
It helps maintain context and coherency throughout the interaction, enabling the AI to generate more relevant 
and accurate responses. Memory, such as ConversationBufferMemory, acts as a wrapper around ChatMessageHistory, 
extracting the messages and providing them to the chain for better context-aware generation.
'''
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
# Start the conversation
conversation.predict(input='Tell me about Yourself')
# Continue the conversation
conversation.predict(input='What can you do?')
conversation.predict(input='How can you help me with data analysis?')
# Display the conversation
print(conversation)
