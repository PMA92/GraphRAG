from multiprocessing import context
from dotenv import load_dotenv
import os
import streamlit as st
import tempfile
from neo4j import GraphDatabase
import json

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs import Neo4jGraph


import neo4j_graphrag.schema 
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

from neo4j import GraphDatabase
from pypdf import PdfReader
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import OpenAIEmbeddings
import re


def embed_all_entities(driver: GraphDatabase.driver):
    LABEL = "Fact"
    TEXT_PROP = "text"
    EMBED_PROP = "embedding"
    BATCH_SIZE = 50
    
    with driver.session() as session:
        rows = session.run(f"""
            MATCH (n:{LABEL})
            WHERE n.{EMBED_PROP} IS NULL
              AND n.{TEXT_PROP} IS NOT NULL
              AND trim(n.{TEXT_PROP}) <> ""
            RETURN elementId(n) AS eid, n.{TEXT_PROP} AS text
        """).data()

    print(f"Found {len(rows)} entities to embed")

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]

        texts = [r["text"] for r in batch]
        vectors = st.session_state["embeddings"].embed_documents(texts)

        with driver.session() as session:
            for r, vec in zip(batch, vectors):
                session.run("""
                    MATCH (n)
                    WHERE elementId(n) = $eid
                    SET n.embedding = $embedding
                """, eid=r["eid"], embedding=vec)

        print(f"Embedded {i + len(batch)} / {len(rows)}")
    

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
    
    response = client.invoke(prompt)
        
    
    res = response.content
    res = re.sub(r"```json|```", "", res).strip()
    print(res)
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
            MERGE (f:Fact {
                text: $source + " " + $relationship + " " + $target
                })
            MERGE (f)-[:ABOUT]->(a)
            MERGE (f)-[:ABOUT]->(b)
            """,
            source=source,
            target=target,
            relationship=relationship,
            database = "neo4j"
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
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        st.session_state["embeddings"] = embeddings
        st.session_state["llm"] = llm
    else:
        embeddings = OpenAIEmbeddings()
        st.session_state["embeddings"] = embeddings
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
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
            qaGraph = Neo4jGraph(
                url=url,
                username=user,
                password=password,
                database="neo4j"     
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
    qaGraph = Neo4jGraph(
                url=url,
                username=st.session_state["user"],
                password=st.session_state["password"],
                database="neo4j"     
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

                embed_all_entities(graph)               

                index = Neo4jVector.from_existing_graph(
                    embedding=st.session_state["embeddings"],
                    username=st.session_state["user"],
                    password=st.session_state["password"],
                    node_label="Fact",
                    url=st.session_state["url"],
                    database="neo4j",
                    text_node_properties=["text"], 
                    embedding_node_property="embedding", 
                    index_name="fact_vector", 
                    search_type="vector" 
                )


                st.success("Uploaded file")
                schema = neo4j_graphrag.schema.get_structured_schema(driver=graph)



        st.subheader("Ask a Question")

        with st.form(key='question_form'):
            question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='Submit')

            docs = index.similarity_search(question, k=5)
            context = "\n".join(d.page_content for d in docs)

            template=ChatPromptTemplate.from_template(template=f"""
                You are answering questions over a graph-backed knowledge base.

                Context:
                {context}

                Question:
                {question}

                If the context directly answers the question, answer using ONLY the context.
                If structured data is required, generate a Cypher query.

                Respond in JSON:
                {{
                  "mode": "answer" | "cypher",
                  "answer": "...",
                  "cypher": "..."
                }}
                """
            )


            print("context from vector: ", docs)
            qa = GraphCypherQAChain.from_llm(
                llm=llm,
                cypher_prompt=template,
                graph=qaGraph,
                verbose=True,
                allow_dangerous_requests=True
            )
            st.session_state['qa'] = qa
            if submit_button and question:
                with st.spinner("Generating answer..."):
                    with graph.session() as session:
                        res = st.session_state['qa'].invoke({
                            "query": question,
                        })
                        st.write("\n**Answer:**\n" + res['result'])

















