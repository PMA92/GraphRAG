from neo4j import GraphDatabase

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j+s://8772a154.databases.neo4j.io"
USER = "neo4j"
PASS = "vIkm6ibcVhIkm5sGZC78ABDXtucOEa0nDHQJMYxTP90"

from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os

load_dotenv()

graph = Neo4jGraph(
    url=URI,
    username=USER,
    password=PASS
)

print(graph.query("RETURN 'Connected!' AS message"))