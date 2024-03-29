"""
Agents in LangChain

In LangChain, agents are high-level components that use language models (LLMs) to determine which actions
to take and in what order. An action can either be using a tool and observing its output or returning it to the user.
Tools are functions that perform specific duties, such as Google Search, database lookups, or Python REPL.

Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation,
and repeating that until done.

Several types of agents are available in LangChain:

The zero-shot-react-description agent uses the ReAct framework to decide which tool to employ based purely on
the tool's description. It necessitates a description of each tool.
The react-docstore agent engages with a docstore through the ReAct framework. It needs two tools: a Search tool and
a Lookup tool. The Search tool finds a document, and the Lookup tool searches for a term in the most recently
discovered document.
The self-ask-with-search agent employs a single tool named Intermediate Answer, which is capable of looking up factual
responses to queries. It is identical to the original self-ask with the search paper, where a Google search API
was provided as the tool.
The conversational-react-description agent is designed for conversational situations. It uses the ReAct framework
to select a tool and uses memory to remember past conversation interactions.
In our example, the Agent will use the Google Search tool to look up recent information about the Mars rover and
generates a response based on this information.

First, you want to set the environment variables “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use Google Search
via API. Refer to this article for a guide on how to get them.
https://python.langchain.com/docs/integrations/tools/google_search
"""


from langchain.llms import OpenAI

from langchain.agents import AgentType
from langchain.agents import initialize_agent

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from config import openai_api_key, active_loop_key, google_api, google_cse

"""
langchain.llms.OpenAI: This is used to create an instance of the OpenAI language model, 
which can generate human-like text based on the input it's given.

langchain.agents.load_tools: This function is used to load a list of tools that an AI agent can use.

langchain.agents.initialize_agent: This function initializes an AI agent that can use a given set of tools 
and a language model to interact with users.

langchain.agents.Tool: This is a class used to define a tool that an AI agent can use. 
A tool is defined by its name, a function that performs the tool's action, and a description of the tool.

langchain.utilities.GoogleSearchAPIWrapper: This class is a wrapper for the Google Search API, allowing it to be used as a tool by an AI agent. It likely contains a method that sends a search query to Google and retrieves the results.
"""

llm = OpenAI(model='text-davinci-003', temperature=0)

search = GoogleSearchAPIWrapper()

# Now create a tool object
"""
The Tool object represents a specific capability or function the system can use. 
In this case, it's a tool for performing Google searches.

It is initialized with three parameters:

name parameter: This is a string that serves as a unique identifier for the tool. In this case, 
the name of the tool is "google-search."

func parameter: This parameter is assigned the function that the tool will execute when called. 
In this case, it's the run method of the search object, which presumably performs a Google search.

description parameter: This is a string that briefly explains what the tool does. 
The description explains that this tool is helpful when you need to use Google to answer questions about current events.
"""
tools = [
    Tool(
        name='google-search',
        func=search.run,
        description='useful for when you need to search google to answer questions about current events'
    )
]

# Next, we create an agent that uses our Google Search tool:
"""
initialize_agent(): This function call creates and initializes an agent. 
An agent is a component that determines which actions to take based on user input. 
These actions can be using a tool, returning a response to the user, or something else.

tools:  represents the list of Tool objects that the agent can use.

agent="zero-shot-react-description": The "zero-shot-react-description" type of an Agent uses the ReAct framework 
to decide which tool to use based only on the tool's description.

verbose=True: when set to True, it will cause the Agent to print more detailed information about what it's doing. 
This is useful for debugging and understanding what's happening under the hood.

max_iterations=6: sets a limit on the number of iterations the Agent can perform before stopping. 
It's a way of preventing the agent from running indefinitely in some cases, which may have unwanted monetary costs.
"""
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=6
)

# now, we can check out the response:
response = agent("What's the latest news about the India's moon rover?")
print(response['output'])

# The example demonstrates initializing and using a "zero-shot-react-description" agent with a Google search tool.