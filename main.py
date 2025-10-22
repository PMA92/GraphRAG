from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph



load_dotenv()

st.set_page_config(
    layout="wide",
    page_title="GraphRAG",
)




left, right, mid, rightmid, right= st.columns([1, 2, 3, 4, 5])


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
    llm = ChatOpenAI(model_name="gpt-4o")
    st.session_state["embeddings"] = embeddings
    st.session_state["llm"] = llm
else:
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model_name="gpt-4o")

if password and url and user:
    st.session_state[url] = url 
    st.session_state[password] = password 
    st.session_state[user] = user 
    try:
        graph = Neo4jGraph(
        url = url,
        password = password,
        username = user
    )
        if graph and llm:
            st.session_state[llm] = llm
            st.session_state[graph] = graph
            submitted = st.form_submit_button("Log In")
            if submitted:
                st.switch_page("mainmenu.py")
    except Exception as e:
        st.error(f"Invalid Neo4J Credentials, check again under error message: {e}")
    














