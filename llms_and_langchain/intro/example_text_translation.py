"""
Text Translation

It is one of the great attributes of Large Language models that enables them to perform multiple tasks just by
changing the prompt. We use the same llmCopy variable as defined before. However, pass a different prompt that
asks for translating the query from a source_language to the target_language.
"""

from config import *
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
translation_prompt = PromptTemplate(input_variables=["source_language", "target_language", "text"], template=translation_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

"""
To use the translation chain, call the predictCopy method with the source language, target language, 
and text to be translated:
"""
source_language = "English"
target_language = "Sanskrit"
text = "Your text here"
translated_text = translation_chain.predict(source_language=source_language, target_language=target_language, text=text)
print(translated_text)