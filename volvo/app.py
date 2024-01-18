import hashlib
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from sddb import init_db, load_questions, qa, vector_search

load_dotenv()


# Function to generate a hash of the password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Initialize session state for authentication status
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

# Replace with your own user credentials
USERNAME = "admin"
PASSWORD = "password"

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

if st.session_state["authentication_status"]:
    if questions:
        [tab_text_search, tab_qa_system, tab_questions] = st.tabs(
            ["Text Search", "QA System", "Candidate Questions"]
        )
    else:
        [tab_text_search, tab_qa_system] = st.tabs(["Text Search", "QA System"])

    with tab_text_search:
        query = st.text_input("---", placeholder="Search for something...")
        submit_button = st.button("Search", key="text_search")
        if submit_button:
            results = vector_search(db, query, top_k=5)
            for r in results:
                score = r.content["score"]
                chunk_data = r.outputs("elements", "chunk")
                metadata = chunk_data["metadata"]
                chunk_message = {}
                chunk_message["score"] = score
                chunk_message["metadata"] = metadata
                txt = chunk_data["txt"]
                st.text(txt)
                st.json(chunk_message)

    with tab_qa_system:
        query = st.text_input("---", placeholder="Ask a question...")
        submit_button = st.button("Search", key="qa")
        if submit_button:
            output, out = qa(db, query, vector_search_top_k=5)
            st.markdown(output.content)

            page_messages = []
            for source in out:
                chunk_data = source.outputs("elements", "chunk")
                metadata = chunk_data["metadata"]
                page_number = metadata["page_number"]
                points = metadata["points"]
                score = source["score"]
                page_messages.append(
                    {"page_number": page_number, "points": points, "score": score}
                )
            df = pd.DataFrame(page_messages)
            st.table(df)
    query = "Reduced wheel brake usage"
    out = vector_search(db, query)

    if questions:
        question_text = "\n\n".join(questions)
        with tab_questions:
            st.markdown(question_text)
