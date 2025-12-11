from neo4j import GraphDatabase

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j+s://8772a154.databases.neo4j.io"
AUTH = ("neo4j", "vIkm6ibcVhIkm5sGZC78ABDXtucOEa0nDHQJMYxTP90")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    driver.execute_query("RETURN 1 AS result")