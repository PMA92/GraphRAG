import streamlit as st
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

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
        apikey = st.text_input("Enter your OpenAI API Key")
        st.form_submit_button("Log In")









