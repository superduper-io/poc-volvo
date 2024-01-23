import hashlib
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from sddb import init_db, load_questions, qa, vector_search
from utils import get_related_documents, get_related_merged_documents

load_dotenv()


# Function to generate a hash of the password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Initialize session state for authentication status
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

# Replace with your own user credentials
# TODO: Move to config file or generate a random password
USERNAME = "admin"
PASSWORD = "yKxYcSVRP9z+xb3jstTtfUq4YP8="

# Hashed password
hashed_password = hash_password(PASSWORD)

# User input for authentication
input_username = st.sidebar.text_input("Username")
input_password = st.sidebar.text_input("Password", type="password")

# Check the username and password
if st.sidebar.button("Login"):
    if (
        input_username == USERNAME
        and hashlib.sha256(input_password.encode()).hexdigest() == hashed_password
    ):
        st.session_state["authentication_status"] = True
        st.sidebar.success("Logged in as {}".format(input_username))
    else:
        st.session_state["authentication_status"] = False
        st.sidebar.error("Incorrect Username/Password")

st.title(os.environ.get("TITLE", "SuperDuperDB"))

db = st.cache_resource(init_db)()
questions = st.cache_resource(load_questions)()


def get_user_input(input_mode, input_key, questions):
    """
    A function to get user input based on the input mode
    """
    if input_mode == "Text Input":
        return st.text_input(
            "Enter your text", placeholder="Type here...", key=input_key
        )
    else:  # Question Selection
        return st.selectbox("Choose a question:", questions, key=input_key)


if st.session_state["authentication_status"]:
    [tab_text_search, tab_qa_system] = st.tabs(["Text Search", "QA System"])

    with tab_text_search:
        search_mode = st.radio(
            "Choose your search mode:",
            ["Question Selection", "Text Input"],
            key="search_mode",
            horizontal=True,
        )
        query = get_user_input(search_mode, "text_search_query", questions)

        submit_button = st.button("Search", key="text_search")
        if submit_button:
            st.markdown("#### Query")
            st.markdown(query)
            results = vector_search(db, query, top_k=5)
            st.markdown("#### Related Documents:")
            for text, img in get_related_merged_documents(results, query):
                st.markdown(text)
                if img:
                    st.image(img)

    with tab_qa_system:
        qa_mode = st.radio(
            "Choose your input mode:",
            ["Question Selection", "Text Input"],
            key="qa_mode",
            horizontal=True,
        )
        query = get_user_input(qa_mode, "qa_query", questions)

        submit_button = st.button("Search", key="qa")
        if submit_button:
            st.markdown("#### Query")
            st.markdown(query)
            output, out = qa(db, query, vector_search_top_k=5)
            st.markdown("#### Answer:")
            st.markdown(output.content)

            st.markdown("#### Related Documents:")
            for text, img in get_related_merged_documents(out, output.content):
                st.markdown(text)
                if img:
                    st.image(img)
