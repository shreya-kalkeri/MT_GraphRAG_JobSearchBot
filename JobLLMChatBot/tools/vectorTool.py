from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from llm import embeddings
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from llm import llm
from langchain.chains import create_retrieval_chain

# Initializing an existing Neo4jVector store
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                                 # to embed user's query
    graph=graph,                                # to interact with Graph RAG database
    index_name="JobSkills",                     # Vector Index name
    node_label="JobTitle",                      # Label of the node which is used to populate the index
    text_node_property="Skills",                # Property containing the original text value 
    embedding_node_property="SkillEmbedding",   # Property containing the embeddings of original text value
    retrieval_query="""
RETURN
    node.Skills AS text,
    score,
    {
        JobTitle: node.JobTitle,
        JobId: node.JobID,
        Source: node.JobPostingURL,
        Location: node.Locations,
        Company: [ (c:Company)-[:OFFERS]->(node) | c.name ],
        AboutCompany: [ (c:Company)-[:OFFERS]->(node) | c.AboutCompany ],
        CompanyTagline: [ (c:Company)-[:OFFERS]->(node) | c.CompanyTagline ],
        EmployeeCount: [ (c:Company)-[:OFFERS]->(node) | c.EmployeeCount ],
        CompanyFounded: [ (c:Company)-[:OFFERS]->(node) | c.CompanyFounded ]
    } AS metadata
"""
)

# Create a Retriever 
retriever = neo4jvector.as_retriever()

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}")
    ]
)

# Create a retrieval chain to insert the documents into prompt, pass this prompt to an LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt) 

plot_retrieval = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# Function to invoke the retrieval chain 
def get_job_skills(input):
    return plot_retrieval.invoke({"input": input})