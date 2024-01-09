"""
Prompt use case:
A key feature of LangChain is its support for prompts, which encompasses prompt management, prompt optimization,
and a generic interface for all LLMs. The framework also provides common utilities for working with LLMs.

ChatPromptTemplate is used to create a structured conversation with the AI model, making it easier to manage the
flow and content of the conversation. In LangChain, message prompt templates are used to construct and work with
prompts, allowing us to exploit the underlying chat model's potential fully.

System and Human prompts differ in their roles and purposes when interacting with chat models.
SystemMessagePromptTemplate provides initial instructions, context, or data for the AI model, while
HumanMessagePromptTemplate are messages from the user that the AI model responds to.
"""

# To illustrate it, let’s create a chat-based assistant that helps users find information about movies.

from config import *

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

chat = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

template = "You are an assistant that helps users find information about movies."

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Find information about the movie {movie_title}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

response = chat(chat_prompt.format_prompt(movie_title='Inception').to_messages())

print(response.content)

"""
Using the to_messages object in LangChain allows you to convert the formatted value of a chat prompt template into 
a list of message objects. This is useful when working with chat models, as it provides a structured way to manage 
the conversation and ensures that the chat model can understand the context and roles of the messages.
"""