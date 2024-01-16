import hashlib
import os
import pandas as pd

import streamlit as st

from superduperdb import CFG, Document, superduper
from superduperdb.backends.mongodb import Collection


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



@st.cache_resource
def _init():
    return superduper(
        mongodb_uri,
        artifact_store="filesystem://data/artifacts/",
        downloads_folder="./data/downloads",
    )


CFG.cluster.backfill_batch_size = 5000

mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/volvo")
db = _init()

st.title("Volvo with SuperDuperDB")

chunk_collection = Collection("_outputs.elements.chunk")


def _vector_search(k, v, index, n=3):
    results = db.execute(
        chunk_collection.like(
            Document({k: v}),
            vector_index=index,
            n=n,
        ).find({})
    )
    return sorted(results, key=lambda x: x.content["score"], reverse=True)


def _qa(query):

    output, out = db.predict(
        model_name='llm',
        input=query,
        context_select=chunk_collection.like(
            Document({"_outputs.elements.chunk": query}),
            vector_index="vector-index",
            n=5,
        ).find({}),
        context_key='_outputs.elements.chunk.0.txt',
    )
    return output, out


if st.session_state["authentication_status"]:
    [tab_text_search, tab_qa_system] = st.tabs(["Text Search", "QA System"])

    with tab_text_search:
        query = st.text_input("---", placeholder="Search for something...")
        submit_button = st.button("Search", key='text_search')
        if submit_button:
            results = _vector_search(
                k="_outputs.elements.chunk", v=query, index="vector-index", n=5
            )
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
        submit_button = st.button("Search", key='qa')
        if submit_button:
            output, out = _qa(query)
            st.markdown(output.content)

            page_messages = []
            for source in sorted(out, key=lambda x:x.content['score'], reverse=True):
                chunk_data = source.outputs('elements', 'chunk')
                metadata = chunk_data['metadata']
                page_number = metadata['page_number']
                points = metadata['points']
                score = source["score"]
                page_messages.append( { "page_number": page_number, "points": points, "score": score } )
            df = pd.DataFrame(page_messages)
            st.table(df)
