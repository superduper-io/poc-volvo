import os
import click

import sentence_transformers
from dotenv import load_dotenv
from tqdm import tqdm
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
from superduperdb import logging
from superduperdb.base.artifact import Artifact
from superduperdb.ext.openai import OpenAIEmbedding

from utils import get_chunks

load_dotenv()


SOURCE_KEY = "elements"
COLLECTION_NAME_SOURCE = "source"

MODEL_IDENTIFIER_CHUNK = "chunk"
MODEL_IDENTIFIER_LLM = "llm"
MODEL_IDENTIFIER_EMBEDDING = "text-embedding-ada-002"
VECTOR_INDEX_IDENTIFIER = "vector-index"

COLLECTION_NAME_CHUNK = f"_outputs.{SOURCE_KEY}.{MODEL_IDENTIFIER_CHUNK}"
CHUNK_OUTPUT_KEY = f"_outputs.{SOURCE_KEY}.{MODEL_IDENTIFIER_CHUNK}"


def _predict(self, X, one: bool = False, **kwargs):
    if isinstance(X, str):
        if isinstance(self.preprocess, Artifact):
            X = self.preprocess.artifact(X)
        return self._predict_one(X)

    if isinstance(self.preprocess, Artifact):
        X = [self.preprocess.artifact(i) for i in X]

    out = []
    batch_size = kwargs.pop("batch_size", 100)
    for i in tqdm(range(0, len(X), batch_size)):
        out.extend(self._predict_a_batch(X[i : i + batch_size], **kwargs))
    return out


OpenAIEmbedding._predict = _predict


def init_db():
    mongodb_uri = os.getenv("MONGODB_URI", "superduperdb-demo")
    artifact_store = os.getenv("ARTIFACT_STORE", "data/artifact_store")

    db = superduper(mongodb_uri, artifact_store=f"filesystem://{artifact_store}")
    return db


def save_pdfs(db, pdf_folder):
    from superduperdb import Document
    from superduperdb.ext.unstructured.encoder import unstructured_encoder
    from pdf2image import convert_from_path

    db.add(unstructured_encoder)

    pdf_names = [pdf for pdf in os.listdir(pdf_folder) if pdf.endswith(".pdf")]

    pdf_paths = [os.path.join(pdf_folder, pdf) for pdf in pdf_names]
    collection = Collection(COLLECTION_NAME_SOURCE)
    to_insert = [
        Document({"elements": unstructured_encoder(pdf_path)}) for pdf_path in pdf_paths
    ]
    db.execute(collection.insert_many(to_insert))

    logging.info(f"Converting {len(pdf_paths)} pdfs to images")
    image_folders = os.environ.get("IMAGES_FOLDER", "data/pdf-images")
    for pdf_name in pdf_names:
        pdf_path = os.path.join(pdf_folder, pdf_name)
        print(pdf_path)
        image_folder = os.path.join(image_folders, pdf_name)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        with tqdm(dynamic_ncols=True) as pbar:
            page_num = 0
            batch_size = 5
            while True:
                images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num + batch_size)
                if not images:
                    pbar.close()
                    break

                for image in images:
                    image.save(os.path.join(image_folder, f"{page_num}.jpg"))
                    page_num += 1
                    pbar.update(1)


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


def add_vector_search_model(db, use_openai=False):
    chunk_collection = Collection("_outputs.elements.chunk")

    if use_openai:
        from superduperdb.ext.openai import OpenAIEmbedding
        from superduperdb.base.artifact import Artifact

        def preprocess(x):
            if isinstance(x, dict):
                # For model chains, the logic of this key needs to be optimized.
                chunk = sorted(x.items())[-1][1]
                return chunk["txt"]
            return x

        # Create an instance of the OpenAIEmbedding model with the specified identifier ('text-embedding-ada-002')
        model = OpenAIEmbedding(
            identifier=MODEL_IDENTIFIER_EMBEDDING,
            model="text-embedding-ada-002",
            preprocess=Artifact(preprocess),
        )
    else:

        def preprocess(x):
            if isinstance(x, dict):
                # For model chains, the logic of this key needs to be optimized.
                chunk = sorted(x.items())[-1][1]
                return "passage: " + chunk["txt"]
            return "query: " + x

        model = Model(
            identifier=MODEL_IDENTIFIER_EMBEDDING,
            object=sentence_transformers.SentenceTransformer(
                "intfloat/multilingual-e5-large"
            ),
            encoder=vector(shape=(1024,)),
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
    logging.info(f"Vector search query: {query}")
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
    logging.info(f"QA query: {query}")
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
    use_openai = os.getenv("USE_OPENAI").upper() == "TRUE"
    save_pdfs(db, "pdf-folders")
    add_chunk_model(db)
    add_vector_search_model(db, use_openai=use_openai)
    add_llm_model(db, use_openai=use_openai)


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
