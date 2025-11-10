from dotenv import load_dotenv
import os
import streamlit as st
import tempfile
from openai import OpenAI
from neo4j import GraphDatabase

from openai import OpenAI
from neo4j import GraphDatabase
from pypdf import PdfReader


def load_pages_from_pdf(doc):
    loader = PdfReader(doc)
    pages = loader.pages
    content = ""
    for page in pages:
        content.append(page.extract_text())
    return content

def documents_to_graph_elements(docs):
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
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": prompt
        }
        ]
    )
    return client.choices[0].message.content




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
        llm = OpenAI(model_name="gpt-4o")
        st.session_state["embeddings"] = embeddings
        st.session_state["llm"] = llm
    else:
        embeddings = OpenAIEmbeddings()
        llm = OpenAI(model_name="gpt-4o")
    if password and url and user:
        st.session_state["url"] = url 
        st.session_state["password"] = password 
        st.session_state["user"] = user 
        try:
            graph = GraphDatabase(
            url = url,
            password = password,
            username = user
        )

            if graph and llm:
                st.session_state["llm"] = llm
                st.session_state["graph"] = graph
                if sub:
                    switch_screen("menu")

        except Exception as e:
            st.error(f"Invalid Neo4J Credentials, check again under error message: {e}")


elif st.session_state["screen"] == "menu":
    st.title("GraphRAG")
    st.write("Here you will upload PDFs and make queries.")
    graph = st.session_state["graph"]
    llm = st.session_state["llm"]
    # Example content
    uploaded_file = st.file_uploader("Upload pdf to knowledge base here", type="pdf")
    if uploaded_file:
        with st.spinner("Uploading file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

                loader = PdfReader(tmp_file_path)
                pages = loader.load_and_split()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
                docs = text_splitter.split_documents(pages)

                lc_docs = []
                for doc in docs:
                    lc_docs.append(Document(page_content=doc.page_content.replace("\n", ""), 
                    metadata={'source': uploaded_file.name}))

                # Clear the graph database
                cypher = """
                  MATCH (n)
                  DETACH DELETE n;
                """
                graph.query(cypher)
                transformer = LLMGraphTransformer(
                    llm=llm,
                    node_properties=True,
                    relationship_properties=True
                )

                graph_documents = transformer.convert_to_graph_documents(lc_docs)

                graph.add_graph_documents(graph_documents)

                index = Neo4jVector(
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
                )

                st.success("Uploaded file")

                schema = graph.get_schema()

                template="""
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

                question_prompt = ChatPromptTemplate(
                    template=template, 
                    input_variables=["schema", "question"] 
                )

                qa = GraphCypherQAChain.from_llm(
                    llm=llm,
                    graph=graph,
                    cypher_prompt=question_prompt,
                    verbose=True,
                    allow_dangerous_requests=True
                )
                st.session_state['qa'] = qa

        if 'qa' in st.session_state:
            st.subheader("Ask a Question")
            with st.form(key='question_form'):
                question = st.text_input("Enter your question:")
                submit_button = st.form_submit_button(label='Submit')
            if submit_button and question:
                with st.spinner("Generating answer..."):
                    res = st.session_state['qa'].invoke({"query": question})
                    st.write("\n**Answer:**\n" + res['result'])
















