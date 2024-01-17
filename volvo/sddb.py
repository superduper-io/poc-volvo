import os

import sentence_transformers
from dotenv import load_dotenv
from superduperdb import (
    Document,
    Listener,
    Model,
    Schema,
    VectorIndex,
    superduper,
    vector,
)
from superduperdb.backends.mongodb import Collection

from utils import get_chunks

load_dotenv()


SOURCE_KEY = "elements"
COLLECTION_NAME_SOURCE = "source"

MODEL_IDENTIFIER_CHUNK = "chunk"
MODEL_IDENTIFIER_embedding = "embedding"
MODEL_IDENTIFIER_LLM = "llm"
VECTOR_INDEX_IDENTIFIER = "vector-index"

COLLECTION_NAME_CHUNK = f"_outputs.{SOURCE_KEY}.{MODEL_IDENTIFIER_CHUNK}"
CHUNK_OUTPUT_KEY = f"_outputs.{SOURCE_KEY}.{MODEL_IDENTIFIER_CHUNK}"


def init_db(db_name="volvo"):
    mongodb_uri = os.getenv("MONGODB_URI").rstrip("/") + "/" + db_name
    artifact_store = os.getenv("ARTIFACT_STORE")

    db = superduper(mongodb_uri, artifact_store=f"filesystem://{artifact_store}")
    return db


def save_pdfs(db, pdf_folder):
    from superduperdb import Document
    from superduperdb.ext.unstructured.encoder import unstructured_encoder

    db.add(unstructured_encoder)

    pdf_paths = [os.path.join(pdf_folder, pdf) for pdf in os.listdir(pdf_folder)]
    collection = Collection(COLLECTION_NAME_SOURCE)
    to_insert = [
        Document({"elements": unstructured_encoder(pdf_path)}) for pdf_path in pdf_paths
    ]
    db.execute(collection.insert_many(to_insert))


def add_chunk_model(db):
    chunk_model = Model(
        identifier=MODEL_IDENTIFIER_CHUNK,
        object=get_chunks,
        flatten=True,
        model_update_kwargs={"document_embedded": False},
        output_schema=Schema(identifier="myschema", fields={"txt": "string"}),
    )

    db.add(
        Listener(
            model=chunk_model,  # Assuming video2images is your SuperDuperDB model
            select=Collection(COLLECTION_NAME_SOURCE).find(),
            key="elements",
        )
    )


def add_vector_search_model(db):
    chunk_collection = Collection("_outputs.elements.chunk")

    model = Model(
        identifier=MODEL_IDENTIFIER_embedding,
        object=sentence_transformers.SentenceTransformer("BAAI/bge-large-en-v1.5"),
        encoder=vector(shape=(384,)),
        predict_method="encode",  # Specify the prediction method
        preprocess=lambda x: x["0"]["txt"] if isinstance(x, dict) else x,
        postprocess=lambda x: x.tolist(),  # Define postprocessing function
        batch_predict=True,  # Generate predictions for a set of observations all at once
    )

    db.add(
        VectorIndex(
            # Use a dynamic identifier based on the model's identifier
            identifier=VECTOR_INDEX_IDENTIFIER,
            # Specify an indexing listener with MongoDB collection and model
            indexing_listener=Listener(
                select=chunk_collection.find(),
                key=CHUNK_OUTPUT_KEY,  # Key for the documents
                model=model,  # Specify the model for processing
                predict_kwargs={"max_chunk_size": 1000},
            ),
        )
    )


def vector_search(db, query, top_k=5):
    query = "Reduced wheel brake usage"
    collection = Collection(COLLECTION_NAME_CHUNK)
    out = db.execute(
        collection.like(
            Document({CHUNK_OUTPUT_KEY: query}),
            vector_index=VECTOR_INDEX_IDENTIFIER,
            n=top_k,
        ).find({})
    )
    return out


def add_llm_model(db, use_openai=False):
    # Define the prompt for the llm model
    prompt_template = (
        "The following is a document and question about the volvo user manual\n"
        "Only provide a very concise answer\n"
        "{context}\n\n"
        "Here's the question:{input}\n"
        "answer:"
    )

    if use_openai:
        from superduperdb.ext.llm.openai import OpenAI

        llm = OpenAI(identifier=MODEL_IDENTIFIER_LLM, prompt_template=prompt_template)

    else:
        from superduperdb.ext.llm.vllm import VllmModel

        llm = VllmModel(
            identifier=MODEL_IDENTIFIER_LLM,
            model_name="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            prompt_template=prompt_template,
            vllm_kwargs={"max_model_len": 2048, "quantization": "awq"},
            inference_kwargs={"max_tokens": 2048},
        )
    # Add the llm instance
    db.add(llm)


def qa(db, query, vector_search_top_k=5):
    collection = Collection(COLLECTION_NAME_CHUNK)
    output, out = db.predict(
        model_name=MODEL_IDENTIFIER_LLM,
        input=query,
        context_select=collection.like(
            Document({CHUNK_OUTPUT_KEY: query}),
            vector_index=VECTOR_INDEX_IDENTIFIER,
            n=vector_search_top_k,
        ).find({}),
        context_key=f"{CHUNK_OUTPUT_KEY}.0.txt",
    )
    return output, out


def build():
    db = init_db()
    save_pdfs(db, "pdf-folders")
    add_chunk_model(db)
    add_vector_search_model(db)
    add_llm_model(db, use_openai=os.getenv("USE_OPENAI").upper() == "TRUE")

if __name__ == "__main__":
    build()
