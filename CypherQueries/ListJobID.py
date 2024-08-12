from neo4j import GraphDatabase

# Connect to the Neo4j database
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def get_job_ids():
    with driver.session() as session:
        result = session.run("MATCH (j:JobTitle) RETURN j.JobID AS JobID")
        job_ids = [record["JobID"] for record in result]
    return job_ids

# Fetch the JobID list
job_ids = get_job_ids()
print(job_ids)