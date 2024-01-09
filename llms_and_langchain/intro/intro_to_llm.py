from config import *

## LLMs in general:
"""
LLMs are deep learning models with billions of parameters that excel at a wide range of natural language processing 
tasks. They can perform tasks like translation, sentiment analysis, and chatbot conversations without being 
specifically trained for them. LLMs can be used without fine-tuning by employing "prompting" techniques, 
where a question is presented as a text prompt with examples of similar problems and solutions.

Architecture:
LLMs typically consist of multiple layers of neural networks, feedforward layers, embedding layers, and 
attention layers. These layers work together to process input text and generate output predictions.

Maximum number of tokens:
In the LangChain library, the LLM context size, or the maximum number of tokens the model can process, 
is determined by the specific implementation of the LLM. In the case of the OpenAI implementation in LangChain, 
the maximum number of tokens is defined by the underlying OpenAI model being used. To find the maximum number of 
tokens for the OpenAI model, refer to the max_tokensCopy attribute provided on the OpenAI documentation or API. 

For example, if you’re using the GPT-3Copy model, the maximum number of tokens supported by the model is 2,049. 
The max tokens for different models depend on the specific version and their variants. 
(e.g., davinciCopy, curieCopy, babbageCopy, or adaCopy) Each version has different limitations, with higher versions 
typically supporting larger number of tokens.

Note:-
It is important to ensure that the input text does not exceed the maximum number of tokens supported by the model, 
as this may result in truncation or errors during processing. To handle this, you can split the input text into 
smaller chunks and process them separately, making sure that each chunk is within the allowed token limit. 
You can then combine the results as needed.

Example:
Here's an example of how you might handle text that exceeds the maximum token limit for a given LLM in LangChain. 
Mind that the following code is partly pseudocode. It's not supposed to run, but it should give you the idea of 
how to handle texts longer than the maximum token limit.
"""

from langchain.llms import OpenAI

# Initialize the LLM
# llm = OpenAI(model='text-davinci-003')

# Define the input text
# input_text = "your_long_input_text"

# Determine the maximum number of tokens from documentation
# max_tokens = 4097

# Split the input text into chunks based on the max tokens
# text_chunks = split_text_into_chunks(input_text, max_tokens)

# Process each chunk separately
# results = []
# for chunk in text_chunks:
#     result = llm.process(chunk)
#     results.append(result)

# Combine the results as needed
# final_result = combine_results(results)

# Note that splitting into multiple chunks can hurt the coherence of the text.

## Tokens Distributions and Predicting the Next Token
"""
Large language models like GPT-3 and GPT-4 are pretrained on vast amounts of text data and learn to predict the 
next token in a sequence based on the context provided by the previous tokens. GPT-family models use Causal Language 
modeling, which predicts the next token while only having access to the tokens before it. 
This process enables LLMs to generate contextually relevant text.

The following code uses LangChain’s OpenAICopy class to load GPT-3’s Davinci variation using text-davinci-003
key to complete the sequence, which results in the answer.
"""

# llm = OpenAI(model_name="text-davinci-003", temperature=0)

# text = "What would be a good company name for a company that makes colorful socks?"

# print(llm(text))

## Tracking Token Usage
"""You can use the LangChain library's callback mechanism to track token usage."""

# from langchain.callbacks import get_openai_callback
#
# llm = OpenAI(model='text-davinci-003', n=2, best_of=2)
#
# with get_openai_callback() as cb:
#     result = llm("tell me a joke")
#     print(cb)


## Few-shot learning

"""
Few-shot learning is a remarkable ability that allows LLMs to learn and generalize from limited examples. 
Prompts serve as the input to these models and play a crucial role in achieving this feature. 
With LangChain, examples can be hard-coded, but dynamically selecting them often proves more powerful, 
enabling LLMs to adapt and tackle tasks with minimal training data swiftly.

This approach involves using the FewShotPromptTemplate class, which takes in a PromptTemplate
and a list of a few shot examples. The class formats the prompt template with a few shot examples, 
which helps the language model generate a better response. We can streamline this process by utilizing 
LangChain's FewShotPromptTemplate to structure the approach:
"""

from langchain import PromptTemplate
from langchain import FewShotPromptTemplate

# create our examples
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
]

# create an example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few-shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=['query'],
    example_separator="\n\n"
)

# After creating a template, we pass the example and user query, and we get the results

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# load the model
chat = ChatOpenAI(model_name='gpt-4', temperature=0)
chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
print(chain.run("What's the meaning of life?"))


## Emergent abilities, Scaling laws, and hallucinations

"""
Another aspect of LLMs is their emergent abilities, which arise as a result of extensive pre-training on vast datasets. 
These capabilities are not explicitly programmed but emerge as the model discerns patterns within the data. 
LangChain models capitalize on these emergent abilities by working with various types of models, such as chat models 
and text embedding models. This allows LLMs to perform diverse tasks, from answering questions to generating text 
and offering recommendations.

Lastly, scaling laws describe the relationship between model size, training data, and performance. Generally, 
as the model size and training data volume increase, so does the model's performance. However, this improvement 
is subject to diminishing returns and may not follow a linear pattern. It is essential to weigh the trade-off 
between model size, training data, performance, and resources spent on training when selecting and fine-tuning 
LLMs for specific tasks.

While Large Language Models boast remarkable capabilities but are not without shortcomings, one notable limitation 
is the occurrence of hallucinations, in which these models produce text that appears plausible on the surface but 
is actually factually incorrect or unrelated to the given input. Additionally, LLMs may exhibit biases originating 
from their training data, resulting in outputs that can perpetuate stereotypes or generate undesired outcomes.
"""