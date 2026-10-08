from dotenv import load_dotenv
import hashlib
import os
import streamlit as st
import tempfile
from neo4j import GraphDatabase
from pydantic import BaseModel

import anthropic

import neo4j_graphrag.schema 
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.types import RetrieverResultItem

from pypdf import PdfReader
import re


MODEL="claude-opus-5-5"
EMBED_MODEL = "all-MiniLM-L6-v2"

EMBED_DIMS = 384
INDEX_NAME = "fact_embedding_index"
NEO4J_DATABASE = None

class Triple(BaseModel):
    Source: str
    Relationship: str
    Target: str

class Triples(BaseModel):
    triples:list[Triple]

def embed_all_entities(driver: GraphDatabase.driver):
    LABEL = "Fact"
    TEXT_PROP = "text"
    EMBED_PROP = "embedding"
    BATCH_SIZE = 50
    
    with driver.session() as session:
        rows = session.run(f"""
            MATCH (n:{LABEL})
            WHERE (n.{EMBED_PROP} IS NULL OR size(n.{EMBED_PROP}) <> $dims)
              AND n.{TEXT_PROP} IS NOT NULL
              AND trim(n.{TEXT_PROP}) <> ""
            RETURN elementId(n) AS eid, n.{TEXT_PROP} AS text
        """, dims=EMBED_DIMS).data()

    print(f"Found {len(rows)} entities to embed")

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]

        vectors = [st.session_state["embeddings"].embed_query(r["text"]) for r in batch]

        with driver.session() as session:
            for r, vec in zip(batch, vectors):
                session.run("""
                    MATCH (n)
                    WHERE elementId(n) = $eid
                    SET n.embedding = $embedding
                """, eid=r["eid"], embedding=vec, dims=EMBED_DIMS)

        print(f"Embedded {i + len(batch)} / {len(rows)}")

def fact_formatter(record):
    return RetrieverResultItem(content=record["node"]["text"], metadata={"score": record["score"]})

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

    Documents:
    {docs}
    """
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        output_format=Triples,
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason == "refusal":
        st.error("The model refused to generate a response.")
    if response.parsed_output is None:
        raise RuntimeError("Extraction got cut off - doc could be too large")
    return response.parsed_output.triples

def answer_question(client: anthropic.Anthropic, context: str, question: str):
    prompt = f"""
    Context:
    {context}


    ONLY ANSWER THE QUESTION USING THE CONTEXT GIVEN!
    If the context can't answer it, or give insight to the question in any way, state so.
    Question:
    {question}
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason == "refusal":
        st.error("The model refused to generate a response.")
        return[]
    return "".join(b.text for b in response.content if b.type == "text")
        

def build_graph_nodes_and_relationships(relation_input, graph: GraphDatabase.driver):
    graph.verify_connectivity()

    for item in relation_input:
        source = item.Source
        relationship = item.Relationship
        target = item.Target


        graph.execute_query(
            """
            MERGE (a:Entity {name: $source})
            MERGE (b:Entity {name: $target})
            MERGE (f:Fact {
                text: $source + " " + $relationship + " " + $target
                })
            MERGE (f)-[:ABOUT]->(a)
            MERGE (f)-[:ABOUT]->(b)
            """,
            source=source,
            target=target,
            relationship=relationship,
            database_=NEO4J_DATABASE
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
            llm = st.selectbox("Pick An LLM", ["Claude"]) #add ollama in the future for local reading
            apikey = st.text_input("Enter your API Key")
            sub = st.form_submit_button("Log In")

    if apikey:
        st.session_state["llm"] = anthropic.Anthropic(api_key=apikey)
        st.success("Anthropic API Key set successfully.")
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"] = SentenceTransformerEmbeddings(model=EMBED_MODEL)
    llm = st.session_state.get("llm")
    if sub and password and url and user:
        st.session_state["url"] = url
        st.session_state["password"] = password
        st.session_state["user"] = user
        try:
            graph = GraphDatabase.driver(uri=url, auth=(user, password))
            graph.verify_connectivity()   # actually tests the credentials
            if llm:
                st.session_state["graph"] = graph
                switch_screen("menu")
                st.rerun()                # show the menu now instead of on the next click
            else:
                st.error("Enter your Anthropic API key.")
        except Exception as e:
            st.error(f"Invalid Neo4J Credentials, check again under error message: {e}")


if st.session_state["screen"] == "menu":
    st.title("GraphRAG")
    st.write("Here you will upload PDFs and make queries.")
    graph = st.session_state["graph"]
    llm = st.session_state["llm"]
    uploaded_file = st.file_uploader("Upload pdf to knowledge base here", type="pdf")
    if uploaded_file:
        # Streamlit reruns this script on every click; only process a file once
        file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if st.session_state.get("processed_file") != file_hash:
            with st.spinner("Uploading file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                lc_docs = load_pages_from_pdf(tmp_file_path)
                graph_documents = documents_to_graph_elements(lc_docs, llm)
                build_graph_nodes_and_relationships(graph_documents, graph)
                embed_all_entities(graph)
                create_vector_index(
                    graph,
                    INDEX_NAME,
                    label="Fact",
                    embedding_property="embedding",
                    dimensions=EMBED_DIMS,
                    similarity_fn="cosine",
                    neo4j_database=NEO4J_DATABASE,
                )
                st.session_state["retriever"] = VectorRetriever(
                    graph,
                    index_name=INDEX_NAME,
                    embedder=st.session_state["embeddings"],
                    return_properties=["text"],
                    result_formatter=fact_formatter,
                    neo4j_database=NEO4J_DATABASE,
                )
                st.session_state["processed_file"] = file_hash
            st.success("Uploaded file")

        retriever = st.session_state["retriever"]

        st.subheader("Ask a Question")

        with st.form(key='question_form'):
            question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='Submit')

            if submit_button and question:
                with st.spinner("Generating answer..."):
                    results = retriever.search(query_text=question, top_k=5)
                    context = "\n".join(item.content for item in results.items)
                    out = answer_question(llm, context, question)
                    st.write("\n**Answer:**\n" + out)