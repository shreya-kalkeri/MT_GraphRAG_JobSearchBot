from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from llm import llm
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from graph import graph
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_session_id
from tools.vectorTool import get_job_skills
from tools.cypherTool import graphCypher_qa

# Create a job chat chain
chat_prompt = ChatPromptTemplate.from_messages(
    [
        #("system", "You are an expert job search assistant specialized in providing detailed and accurate information about available job postings. Your goal is to help users find the most relevant job opportunities based on their queries. Answer concisely, ensure clarity, and provide guidance or suggestions when necessary. If specific job data is not available, offer general advice on job searching and related topics."),
        ("system", "You are an expert job search assistant with access to a database of job postings. Your primary task is to retrieve and provide specific job listings or relevant information based on the user's query. Always prioritize retrieving data from the database. If the requested information is not available, only then provide general advice or related information. Do not give general definitions unless explicitly asked."),
        ("human", "{input}"),
    ]
)

job_chat = chat_prompt | llm | StrOutputParser()

# Create multiple tools to complete various tasks 
# Tool 1: 'General Chat' for general text generation
# Tool 2: Tool uses Vector Search Index to recommend jobs
# Tool 3: Tool to convert user question into cypher query
tools = [
    Tool.from_function(
        name="General Chat",
        description="For job search chat not covered by other tools",
        func=job_chat.invoke,
    ),
    Tool.from_function(
        name="Job Search Search",
        description="For when you need to find information about jobs based on a skills",
        func=get_job_skills,
    ),
    Tool.from_function(
        name="Job Information",
        description="Provide information about jobs question using Cypher",
        func=graphCypher_qa
    ) 
]

# Neo4jChatMessageHistory is used to store and retrieve messages from Neo4j 'GraphRAG'
# Create chat history callback function to return memory component.
# Agent passes 'session_id' to retrieve conversation for that specific session
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Agent Prompt
agent_prompt = PromptTemplate.from_template("""
You are a job search expert providing specializing in providing comprehensive and accurate information about job postings, companies, and required skills.
                                            
Your objectives:
Be as helpful as possible by retrieving and presenting detailed information.
Stick strictly to the context provided; do not use pre-trained knowledge outside of the given tools and data.
If the user's request is unclear, ask for clarification instead of assuming.
Do not answer any questions that do not relate to Job, Company and Skill.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```                                            
Additional Guidelines:
Prioritize concise and relevant information, focusing on answering the user's query directly.
If multiple tools are necessary, use them sequentially, ensuring each step is clear.
Use past conversation history to maintain context and continuity in your responses.                                            

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

# Create an Agent 
agent = create_react_agent(llm, tools, agent_prompt)

# Create Agent Executor for executions
agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True
)

# Create a chat agent which is a wrapper to the Agent 
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key = "input",
    history_messages_key = "chat_history"
)

# Create a handler function to call the Conversational agent inorder to respond to the user query

def generate_response(user_input):
    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)
    return response['output']