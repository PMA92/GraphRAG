import streamlit as st
from dotenv import load_dotenv

load_dotenv()

login = st.Page("login.py", title="Login")
mainmenu = st.Page("mainmenu.py", title="Main Menu")

pg = st.navigation([login, mainmenu])

pg.run()

