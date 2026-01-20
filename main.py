from dotenv import load_dotenv
import os
import streamlit as st
import tempfile
from openai import OpenAI
from neo4j import GraphDatabase
import json
from openai import OpenAI

import neo4j_graphrag.schema 
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

from neo4j import GraphDatabase
from pypdf import PdfReader
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import OpenAIEmbeddings
import re

def load_pages_from_pdf(doc):
    loader = PdfReader(doc)
    pages = loader.pages
    content = []
    for page in pages:
        text = page.extract_text()
        if text:
            content.append(text)
    return "\n".join(content)

def documents_to_graph_elements(docs, client):
    prompt = f"""
    Extract knowledge graph triples for each document in the list of docouemnts.

    Format ONLY as a list of JSONs with this relatinoship:
    [
        {{"Source": "...", "Relationship": "...", "Target": "..."}},
    ]

    Documents:
    {docs}
    """
    
    client = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": prompt
            }],
    )
    
    res = client.choices[0].message.content
    res = re.sub(r"```json|```", "", res).strip()
    info = json.loads(res)
    return info


        

def build_graph_nodes_and_relationships(relation_input, graph: GraphDatabase.driver):
    for item in relation_input:
        source = item["Source"]
        relationship = item["Relationship"]
        target = item["Target"]


        graph.verify_connectivity()
        graph.execute_query(
            """
            MERGE (a:Entity {name: $source})
            MERGE (b:Entity {name: $target})
            MERGE (a)-[r:RELATIONSHIP {type: $relationship}]->(b)
            """,
            source=source,
            target=target,
            relationship=relationship,
            database = "GraphRAG"
        )
    


load_dotenv()
st.set_page_config(
    layout="wide",
    page_title="GraphRAG",
)

if "screen" not in st.session_state:
    st.session_state["screen"] = "login"

def switch_screen(screen_name: str):
    st.session_state["screen"] = screen_name


left, right, mid, rightmid, right= st.columns([1, 2, 3, 4, 5])

graph = None
llm = None

if st.session_state["screen"] == "login":
    with rightmid:
        st.title("GraphRAG")
        st.header("Log In")
    with rightmid:
        with st.form("Enter Credentials"):
            url = st.text_input("Neo4J Url")
            user = st.text_input("Neo4J Username")
            password = st.text_input("Neo4J Password")
            llm = st.selectbox("Pick An LLM", ["OpenAI"]) #add ollama in the future for local reading
            apikey = st.text_input("Enter your API Key")
            sub = st.form_submit_button("Log In")

    if 'OPENAI_AI_API_KEY' not in st.session_state and apikey != "":
        os.environ['OPENAI_API_KEY'] = apikey
        st.session_state['OPENAI_API_KEY'] = apikey
        st.success("OpenAI API Key set successfully.")
        embeddings = OpenAIEmbeddings()
        llm = OpenAI()
        st.session_state["embeddings"] = embeddings
        st.session_state["llm"] = llm
    else:
        embeddings = OpenAIEmbeddings()
        llm = OpenAI()
    if password and url and user:
        st.session_state["url"] = url 
        st.session_state["password"] = password 
        st.session_state["user"] = user 
        try:
            auth = (user, password)
            graph = GraphDatabase.driver(
            uri = url,
            auth=auth
        )

            if graph and llm:
                st.session_state["llm"] = llm
                st.session_state["graph"] = graph
                if sub:
                    switch_screen("menu")

        except Exception as e:
            st.error(f"Invalid Neo4J Credentials, check again under error message: {e}")


if st.session_state["screen"] == "menu":
    st.title("GraphRAG")
    st.write("Here you will upload PDFs and make queries.")
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))
    url = os.getenv("NEO4J_URL")
    graph = GraphDatabase.driver(
        uri = url,
        auth=auth
    )
    llm = st.session_state["llm"]
    # Example content
    uploaded_file = st.file_uploader("Upload pdf to knowledge base here", type="pdf")
    if uploaded_file:
        with st.spinner("Uploading file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

                lc_docs = load_pages_from_pdf(tmp_file_path)

                # Clear the graph database
                cypher = """
                  MATCH (n)
                  DETACH DELETE n;
                """
                graph_documents = documents_to_graph_elements(lc_docs, llm)

                build_graph_nodes_and_relationships(graph_documents, graph)                

                """index = vector(
                    embedding=st.session_state["embeddings"],
                    username=st.session_state["user"],
                    password=st.session_state["password"],
                    url=st.session_state["url"],
                    database="neo4j",
                    text_node_properties=["id", "text"], 
                    embedding_node_property="embedding", 
                    index_name="vector_index", 
                    keyword_index_name="entity_index", 
                    search_type="hybrid" 
                )"""

                st.success("Uploaded file")
                schema = neo4j_graphrag.schema.get_structured_schema(driver=graph)


                
        st.subheader("Ask a Question")

        with st.form(key='question_form'):
            question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='Submit')

            template=f"""
                Task: Generate a Cypher statement to query the graph database.
                Instructions:
                Use only relationship types and properties provided in schema.
                Do not use other relationship types or properties that are not provided.
                schema:
                {schema}
                Note: Do not include explanations or apologies in your answers.
                Do not answer questions that ask anything other than creating Cypher statements.
                Do not include any text other than generated Cypher statements.
                Question: {question}""" 


            cypher = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": template},
                ],
            )
            st.session_state['cypher'] = cypher.choices[0].message.content

            if submit_button and question:
                with st.spinner("Generating answer..."):
                    with graph.session() as session:
                        result = session.run(st.session_state['cypher'])
                        records = result.data()
                        records_str = json.dumps(records, indent=2)
                        print("Generating answer...")
                    #res = st.session_state['qa'].invoke({"query": question})
                        st.write("\n**Answer:**\n" + records_str)

















