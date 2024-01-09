from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentType, initialize_agent

from config import dataset_path

## an example of reloading an existing vector store and adding more data.

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')


# load the existing Deep Lake dataset and specify the embedding function
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)


### We then recreate our previous agent and ask a question that can be answered only by the last documents added.

# instantiate the wrapper class for GPT3
llm = OpenAI(model='text-davinci-003', temperature=0)

# create a retriever from the db
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(),
)

# instantiate a tool that uses the retriever
tools = [
    Tool(
        name='Retrieval QA System',
        func=retrieval_qa.run,
        description='Useful for answering questions'
    )
]

# create an agent that uses the tool
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

## Let’s now test our agent with a new question.

response = agent.run("When was Michael Jordan born?")
print(response)