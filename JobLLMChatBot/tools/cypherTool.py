from langchain.prompts.prompt import PromptTemplate
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from llm import llm
from graph import graph

CYPHER_GENERATION_TEMPLATE = """
ou are an expert Neo4j Developer translating user questions into Cypher to answer questions about jobs and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Fine Tuning:
"Generate efficient queries by minimizing the amount of data retrieved. Use LIMIT clauses where appropriate to avoid returning large datasets unless necessary."
"Prefer indexed properties in MATCH and WHERE clauses to speed up query execution. Use indexes for properties like unique identifiers or commonly queried attributes."

Schema:
{schema}

Question:
{question}

Example Cypher Queries:

1. To find which company offers a specific job role, a question like "Which companies are currently hiring for software engineers, and where can I find the job postings?"
```
MATCH (c:Company)-[:OFFERS]->(j:JobTitle {JobTitle: "Software Engineer"})
RETURN j.JobTitle, c.name , j.JobPostingURL
```

2. I have expertise in DevOps. What roles are available, which companies are hiring and in which location?
```
MATCH (c:Company)-[:OFFERS]->(j:JobTitle {JobTitle: "DevOps"})
RETURN j.JobTitle, c.name , j.JobPostingURL, j.Locations 
```

3. What companies are hiring for full-stack developers, where can I find the job postings and tell me something about the company?
```
MATCH (c:Company)-[:OFFERS]->(j:JobTitle {JobTitle: "Full Stack Developer"})
RETURN j.JobTitle, c.name , j.JobPostingURL, c.AboutCompany
```
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

# Create GraphCypherQAChain
graphCypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt
)