from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from config import active_loop_key, openai_api_key, dataset_path

# instantiate the LLM and embeddings models

llm = OpenAI(model='text-davinci-003', temperature=0)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

#create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]

text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# create Deep Lake dataset
# (by default, org id is your username)
# my_activeloop_org_id = "nishantsingh"
# my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
#
# dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
# db.add_documents(docs)

## let's create a RetrievalQA chain:

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever()
)

# Next, let's create an agent that uses the RetrievalQA chain as a tool:
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [
    Tool(
        name='Retrieval QA System',
        func=retrieval_qa.run,
        description='Useful for answering questions.'
    ),
]

agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.run("When was Napoleon born?")
print(response)




