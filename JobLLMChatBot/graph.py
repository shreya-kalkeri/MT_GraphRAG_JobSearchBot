import streamlit as st
from langchain_community.graphs import Neo4jGraph

# Create a instance of class Neo4jGraph inorder to Connect to Neo4j database 'GraphRAG'
graph = Neo4jGraph(
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
)