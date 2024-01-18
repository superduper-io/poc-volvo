import os
import click

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
MODEL_IDENTIFIER_EMBEDDING = "embedding"
MODEL_IDENTIFIER_LLM = "llm"
VECTOR_INDEX_IDENTIFIER = "vector-index"

COLLECTION_NAME_CHUNK = f"_outputs.{SOURCE_KEY}.{MODEL_IDENTIFIER_CHUNK}"
CHUNK_OUTPUT_KEY = f"_outputs.{SOURCE_KEY}.{MODEL_IDENTIFIER_CHUNK}"


def init_db():
    mongodb_uri = os.getenv("MONGODB_URI", "superduperdb-demo")
    artifact_store = os.getenv("ARTIFACT_STORE", "data/artifact_store")

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
            model=chunk_model,
            select=Collection(COLLECTION_NAME_SOURCE).find(),
            key="elements",
        )
    )


def add_vector_search_model(db):
    chunk_collection = Collection("_outputs.elements.chunk")

    def preprocess(x):
        if isinstance(x, dict):
            # For model chains, the logic of this key needs to be optimized.
            chunk = sorted(x.items())[-1][1]
            return chunk["txt"]
        return x

    model = Model(
        identifier=MODEL_IDENTIFIER_EMBEDDING,
        object=sentence_transformers.SentenceTransformer("BAAI/bge-large-en-v1.5"),
        encoder=vector(shape=(384,)),
        predict_method="encode",
        preprocess=preprocess,
        postprocess=lambda x: x.tolist(),
        batch_predict=True,
    )

    db.add(
        VectorIndex(
            identifier=VECTOR_INDEX_IDENTIFIER,
            indexing_listener=Listener(
                select=chunk_collection.find(),
                key=CHUNK_OUTPUT_KEY,  # Key for the documents
                model=model,  # Specify the model for processing
                predict_kwargs={"max_chunk_size": 64},
            ),
        )
    )


def vector_search(db, query, top_k=5):
    collection = Collection(COLLECTION_NAME_CHUNK)
    out = db.execute(
        collection.like(
            Document({CHUNK_OUTPUT_KEY: query}),
            vector_index=VECTOR_INDEX_IDENTIFIER,
            n=top_k,
        ).find({})
    )
    if out:
        out = sorted(out, key=lambda x: x.content["score"], reverse=True)
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
    if out:
        out = sorted(out, key=lambda x: x.content["score"], reverse=True)
    return output, out


def generate_questions_from_db(n=20):
    # TODO: Generate questions for showing in the frontend
    db = init_db()
    datas = []

    llm = db.load("model", MODEL_IDENTIFIER_LLM)
    chunks = list(
        db.execute(
            Collection(COLLECTION_NAME_CHUNK).find({}, {"_id": 1, CHUNK_OUTPUT_KEY: 1})
        )
    )
    generate_template = """
Based on the information provided, please formulate one question related to the document excerpt. Answer in JSON format.

**Context**:
{%s}

Using the information above, generate your questions. Your question can be one of the following types: What, Why, When, Where, Who, How. Please respond in the following format:

```json
{
  \"question_type\": \"Type of question, e.g., 'What'\",
  \"question\": \"Your question \",
}
```
"""
    # reset the prompt function
    llm.prompt_func = lambda x: generate_template % x
    datas = []
    import random

    for chunk in chunks:
        text = chunk.outputs("elements", "chunk")["txt"]
        id_ = chunk["_id"]
        datas.append({"id": id_, "text": text})

    random.shuffle(datas)
    questions = []
    for data in datas[: n * 5]:
        text = data["text"]
        try:
            result = llm.predict(text)
            print(result)
            json_result = eval(result)
            # keep the id
            questions.append(
                {
                    "id": str(data["id"]),
                    "question": json_result["question"],
                }
            )
        except Exception as e:
            print(e)
            continue

        if len(questions) >= n:
            break

    for q in questions:
        print(q["question"])

    question_path = os.environ.get("CANDIDATE_QUESTIONS", "questions.txt")
    print(f"save question list to {question_path}")
    with open(question_path, "w") as f:
        for q in questions:
            f.write(q["question"] + "\n")

    return questions


def load_questions():
    question_path = os.environ.get("CANDIDATE_QUESTIONS", "questions.txt")
    if not os.path.exists(question_path):
        return []
    with open(question_path, "r") as f:
        questions = f.readlines()
    return questions


def setup_db():
    init_db()
    # TODO: Support more configurations for building the database
    db = init_db()
    save_pdfs(db, "pdf-folders")
    add_chunk_model(db)
    add_vector_search_model(db)
    add_llm_model(db, use_openai=os.getenv("USE_OPENAI").upper() == "TRUE")


@click.command()
@click.option(
    "--init",
    is_flag=True,
    default=False,
    help="Build the database",
)
@click.option(
    "--questions_num",
    default=0,
    type=int,
    help="number of questions to generate",
)
def main(init, questions_num):
    if init:
        setup_db()
    if questions_num:
        generate_questions_from_db(n=questions_num)


if __name__ == "__main__":
    main()
