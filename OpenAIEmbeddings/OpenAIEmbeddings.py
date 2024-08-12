import os
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase, Result
from openai import OpenAIError
from time import sleep
import pandas as pd
import secrets as st

# Initialize the OpenAIEmbeddings object
api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)

def generate_embeddings(file_name, limit=None):
    
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=('neo4j', 'password')
    )

    driver.verify_connectivity()

    query = """MATCH (j:JobTitle) WHERE j.Skills IS NOT NULL
    RETURN j.JobID AS JobID, j.JobTitle AS JobTitle, j.Skills AS Skills"""
    
    if limit is not None:
        query += f" LIMIT {limit}"

    jobs = driver.execute_query(
        query,
        result_transformer_=Result.to_df
    )

    print(len(jobs))
    print('Connection established')

    embedding_data = []

    for _, n in jobs.iterrows():
        successful_call = False
        while not successful_call:
            try:
                embedding = embeddings.embed_query(f"{n['JobTitle']}: {n['Skills']}")
                successful_call = True
            except OpenAIError as e:
                print(e)
                print("Retrying in 5 seconds...")
                sleep(5)

        print(n['JobTitle'])

        embedding_data.append({"JobID": n['JobID'], "Embedding": embedding})

    embedding_df = pd.DataFrame(embedding_data)
    embedding_df.head()
    embedding_df.to_csv(file_name, index=False)

# Generate Embeddings
generate_embeddings('data/openai-embeddings.csv', limit=None)