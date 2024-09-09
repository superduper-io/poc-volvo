import os
import click

import sentence_transformers
from dotenv import load_dotenv
from tqdm import tqdm
from superduper import (
    Document,
    Listener,
    model,
    Schema,
    VectorIndex,
    superduper,
    vector,
    QueryModel,
)

from superduper import logging
# from superduper.base.artifact import Artifact
from superduper.ext.openai import OpenAIEmbedding
from superduper_sentence_transformers import SentenceTransformer

from utils import get_chunks

load_dotenv()


SOURCE_KEY = "elements"
COLLECTION_NAME_SOURCE = "source"

MODEL_IDENTIFIER_CHUNK = "chunker"
MODEL_IDENTIFIER_LLM = "llm"
MODEL_IDENTIFIER_EMBEDDING = os.getenv("MODEL_IDENTIFIER_EMBEDDING")
VECTOR_INDEX_IDENTIFIER = "vector-index"

COLLECTION_NAME_CHUNK = f"_outputs.{MODEL_IDENTIFIER_CHUNK}"
CHUNK_OUTPUT_KEY = f"_outputs.{MODEL_IDENTIFIER_CHUNK}"


# def _predict(self, X, one: bool = False, **kwargs):
#     if isinstance(X, str):
#         if isinstance(self.preprocess, Artifact):
#             X = self.preprocess.artifact(X)
#         return self._predict_one(X)
#
#     if isinstance(self.preprocess, Artifact):
#         X = [self.preprocess.artifact(i) for i in X]
#
#     out = []
#     batch_size = kwargs.pop("batch_size", 100)
#     for i in tqdm(range(0, len(X), batch_size)):
#         out.extend(self._predict_a_batch(X[i : i + batch_size], **kwargs))
#     return out


# OpenAIEmbedding._predict = _predict


def init_db():
    mongodb_uri = os.getenv("MONGODB_URI", "superduperdb-demo")
    artifact_store = os.getenv("ARTIFACT_STORE", "data/artifact_store")

    db = superduper(mongodb_uri, artifact_store=f"filesystem://{artifact_store}")
    return db


def save_pdfs(db, pdf_folder):
    from superduper import Document
    from superduper.ext.unstructured.encoder import unstructured_encoder
    from pdf2image import convert_from_path

    db.apply(unstructured_encoder)

    pdf_names = [pdf for pdf in os.listdir(pdf_folder) if pdf.endswith(".pdf")]

    pdf_paths = [os.path.join(pdf_folder, pdf) for pdf in pdf_names]
    # collection = Collection(COLLECTION_NAME_SOURCE)
    to_insert = [
        Document({"elements": unstructured_encoder(pdf_path)}) for pdf_path in pdf_paths
    ]
    db[COLLECTION_NAME_SOURCE].insert_many(to_insert).execute()
    # db.execute(collection.insert_many(to_insert))

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
    upstream_listener = Listener(
        model=get_chunks,
        select=db[COLLECTION_NAME_SOURCE].select(), # "source"
        key=SOURCE_KEY, # elements
        uuid=MODEL_IDENTIFIER_CHUNK, #  "chunker"
        identifier=MODEL_IDENTIFIER_CHUNK
    )
    db.apply(upstream_listener)
    # chunk_model = Model(
    #     identifier=MODEL_IDENTIFIER_CHUNK,
    #     object=get_chunks,
    #     flatten=True,
    #     model_update_kwargs={"document_embedded": False},
    #     output_schema=Schema(identifier="myschema", fields={"txt": "string"}),
    # )
    #
    # db.add(
    #     Listener(
    #         model=chunk_model,
    #         select=Collection(COLLECTION_NAME_SOURCE).find(),
    #         key="elements",
    #     )
    # )


def add_vector_search_model(db, use_openai_embed=False):
    chunk_collection = db[COLLECTION_NAME_CHUNK]

    if use_openai_embed:
        from superduper_openai import OpenAIEmbedding
        # from superduper  import Artifact

        def preprocess(x):
            if isinstance(x, dict):
                # For model chains, the logic of this key needs to be optimized.
                chunk = sorted(x.items())[-1][1]
                return chunk["txt"]
            return x

        # Create an instance of the OpenAIEmbedding model with the specified identifier ('text-embedding-ada-002')
        embedding_model = OpenAIEmbedding(
            identifier=MODEL_IDENTIFIER_EMBEDDING,
            model="text-embedding-ada-002"
            # preprocess=Artifact(preprocess),
        )
    else:

        def preprocess(x):
            if isinstance(x, dict):
                # For model chains, the logic of this key needs to be optimized.
                chunk = sorted(x.items())[-1][1]
                return "passage: " + chunk["txt"]
            return "query: " + x

        embedding_model = SentenceTransformer(
            identifier=MODEL_IDENTIFIER_EMBEDDING,
            object=sentence_transformers.SentenceTransformer(
                "BAAI/bge-base-en-v1.5",
                device="cuda"
            ),
            datatype=vector(shape=(768,)),
            device="cuda",
            # predict_method="encode",
            # preprocess=preprocess,
            postprocess=lambda x: x.tolist(),
            predict_kwargs={"show_progress_bar": True},
        )
    vector_index = \
        VectorIndex(
            identifier=VECTOR_INDEX_IDENTIFIER,
            indexing_listener=Listener(
                select=chunk_collection.select(),
                key=CHUNK_OUTPUT_KEY,  # Key for the documents
                model=embedding_model,  # Specify the model for processing
                # predict_kwargs={"max_chunk_size": 64},
                uuid=MODEL_IDENTIFIER_EMBEDDING, # This one goes into db collection key name
            ),
        )
    db.apply(vector_index)

def vector_search(db, query, top_k=5):
    logging.info(f"Vector search query: {query}")
    chunk_collection = db[COLLECTION_NAME_CHUNK]
    out = db.execute(
        chunk_collection.like(
            Document({CHUNK_OUTPUT_KEY: query}),
            vector_index=VECTOR_INDEX_IDENTIFIER,
            n=top_k,
        ).select({})
    )
    if out:
        out = sorted(out, key=lambda x: x['score'], reverse=True)
    return out


def add_llm_model(db, use_openai=False,use_vllm=False):
    # Define the prompt for the llm model
    prompt_template = (
        "The following is a document and question about the volvo user manual\n"
        "Only provide a very concise answer\n"
        "{context}\n\n"
        "Here's the question:{input}\n"
        "answer:"
    )

    if use_openai:
        from superduper_openai import OpenAIChatCompletion
        llm = OpenAIChatCompletion(identifier=MODEL_IDENTIFIER_LLM,  model='gpt-3.5-turbo')

    elif use_vllm:
        from superduper_vllm import VllmModel
        llm = VllmModel(
            identifier=MODEL_IDENTIFIER_LLM,
            model_name="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            prompt_func=prompt_template,
            vllm_kwargs={
                "gpu_memory_utilization": 0.50,
                "max_model_len": 2048,
                "quantization": "awq"
            },
            predict_kwargs={"max_tokens": 1024, "temperature": 0.8},
        )
    else:
        from superduper_anthropic import AnthropicCompletions
        # os.environ["ANTHROPIC_API_KEY"] = ""
        predict_kwargs = {
            "max_tokens": 1024,
            "temperature": 0.8,
        }
        llm = AnthropicCompletions(identifier='llm', model='claude-2.1', predict_kwargs=predict_kwargs)
        # Ad
    # Add the llm instance
    db.apply(llm)

def build_prompt(query, docs):
    prompt_template = (
        "The following is a document and question about the volvo user manual\n"
        "Only provide a very concise answer\n"
        "{context}\n\n"
        "Here's the question:{input}\n"
        "answer:"
    )
    chunks = [doc['_outputs']['chunker']["txt"] for doc in docs]
    context = "\n\n".join(chunks)
    prompt = prompt_template.format(context=context, input=query)
    return prompt

def qa(db, query, vector_search_top_k=5):
    logging.info(f"QA query: {query}")
    chunk_collection = db[COLLECTION_NAME_CHUNK]
    item = {'_outputs.chunker': '<var:query>'}
    vector_search_model = QueryModel(
        identifier="VectorSearch",
        select=chunk_collection.like(
            item,
            vector_index=VECTOR_INDEX_IDENTIFIER,
            n=vector_search_top_k
        ).select(),
        # postprocess=lambda docs: [{"text": doc['_outputs.chunker'], "_source": doc["_source"],"score": doc["score"]} for doc in docs],
        db=db
    )
    out = vector_search_model.predict(query=query)
    if out:
        out = sorted(out, key=lambda x: x["score"], reverse=True)
        prompt= build_prompt(query,out)
        output = db.load("model","llm").predict(prompt)
    return output, out


from pprint import pprint
def generate_questions_from_db(n=20):
    # TODO: Generate questions for showing in the frontend
    db = init_db()
    datas = []

    llm = db.load("model", MODEL_IDENTIFIER_LLM)
    chunks = list(
        db.execute(
            db[COLLECTION_NAME_CHUNK].find({}, {"_id": 1, CHUNK_OUTPUT_KEY: 1})
        )
    )
    generate_template = """
Based on the information provided, please formulate one question related to the document excerpt. Answer in JSON format.

**Context**:
{%s}

Using the information above, generate your questions. Your question can be one of the following types: What, Why, When, Where, Who, How. Please respond in the following format:


{
  \"question_type\": \"Type of question, e.g., 'What'\",
  \"question\": \"Your question \",
}
 
"""
    # reset the prompt function
    prompt = lambda x: generate_template % x
    datas = []
    import random

    for chunk in chunks:
        text = chunk[CHUNK_OUTPUT_KEY]["txt"]
        id_ = chunk["_id"]
        datas.append({"id": id_, "text": text})

    random.shuffle(datas)
    questions = []
    for data in datas[: n * 5]:
        text = data["text"]
        try:
            result = llm.predict(prompt(text))
            print("llm output\n"+result)
            # For Anthropic model adds extra "Here is a possible question based on the context provided:"
            try:
                json_result = eval(result)
            except SyntaxError:
                json_result = eval(result.split("\n",2)[2])
            print('Formated JSON output')
            pprint(json_result)
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
    use_openai_embed = os.getenv("USE_OPENAI_EMBED").upper() == "TRUE"
    save_pdfs(db, "pdf-folders")
    add_chunk_model(db)
    add_vector_search_model(db, use_openai_embed=use_openai_embed)
    add_llm_model(db, use_openai=use_openai,use_vllm=use_openai)


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
