from agent import generate_response
from graph import graph  
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from llm import llm, embeddings  
from utils import write_message
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from tools.vectorTool import get_job_skills
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from utils import get_session_id

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

tools = [
    Tool.from_function(
        name="Job Skill Search",  
        description="For when you need to find information about jobs based on a skills",
        func=get_job_skills 
    )
]

neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                
    graph=graph,
    index_name="JobSkills",
    node_label="JobTitle",
    text_node_property="Skills",
    embedding_node_property="SkillEmbedding",
    retrieval_query="""
    RETURN
        node.JobTitle AS JobTitle,
        node.Locations AS Location,
        node.JobId AS JobId,
        node.Skills AS Skills,
        [ (c:Company)-[:OFFERS]->(node) | c.name ] AS Company,
        [ (c:Company)-[:OFFERS]->(node) | c.AboutCompany ] AS AboutCompany,
        [ (c:Company)-[:OFFERS]->(node) | c.CompanyTagline ] AS CompanyTagline,
        [ (c:Company)-[:OFFERS]->(node) | c.EmployeeCount ] AS EmployeeCount,
        [ (c:Company)-[:OFFERS]->(node) | c.CompanyFounded ] AS CompanyFounded,
        score
    """
)

def calculate_compatibility_score(user_input):
    agent_prompt = PromptTemplate.from_template("""
    You are a job search expert providing score about user input and existing jobs..
    Be as helpful as possible and return only scores in percentage with 2 decimal points for example: XX.xx, 65.38, 23.13, 12.18 etc.
                                                
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

    Begin!

    Previous conversation history:
    {chat_history}

    New input: {input}
    {agent_scratchpad}
    """)

    agent = create_react_agent(llm, tools, agent_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
        )

    chat_agent = RunnableWithMessageHistory(
        agent_executor,
        get_memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    tokens = user_input.split()
    
    if len(tokens) > 16384:
        st.error(f"Input exceeds maximum token length of {16384}. Please shorten your message.")
        return
    try:
        with st.spinner('Thinking...'):
            response = generate_response(user_input)
            text_info = response
            response = chat_agent.invoke(
                {"input": user_input},
                {"configurable": {"session_id": get_session_id()}})
            print(f"TESTY: {response}, {text_info}")
            return response, text_info
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error(f"Full error details: {e}")

# Programming (Python, R), Statistical Analysis, Data Visualization, Machine Learning, Data Cleaning and Preprocessing, SQL, Big Data Tools (Hadoop, Spark), Deep Learning, Data Mining, Communication Skills