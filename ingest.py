import os
import uuid
import chromadb
import click
import torch
import logging

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List

from langchain.docstore.document import Document
from langchain.embeddings import (
    HuggingFaceInstructEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


import json
import hashlib


def files_difference(source_dir: str, new_files: list):
    files_registry = os.path.join(source_dir, ".files-diff")

    open(files_registry, "a+").close()
    with open(files_registry, "r") as f:
        current_reg = set(f.readlines())

    temp_reg = {hashlib.md5(open(fn, "rb").read()).hexdigest(): fn for fn in new_files}

    new_reg = current_reg.difference(set(temp_reg.keys()))
    with open(files_registry, "a+") as f:
        f.writelines(new_reg)

    return [fn for byte_sum, fn in temp_reg.items() if byte_sum in new_reg]


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory
    all_files = os.listdir(source_dir)
    paths = []
    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1]
        source_file_path = os.path.join(source_dir, file_path)
        if file_extension in DOCUMENT_MAP.keys():
            paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs


def add_document_to_db():
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)


from langchain.document_loaders import YoutubeLoader


# --------------------------------------------------------------
# Load a video transcript from YouTube
# --------------------------------------------------------------
def load_transcript_from_yt() -> List[Document]:
    video_url = "https://www.youtube.com/watch?v=riXpu1tHzl0"
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()[0]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
    docs = text_splitter.split_documents(transcript)
    return docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def load_vector_db(documents: List[Document], embeddings, ids: List[str] = []):
    """ """
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client=chromadb.PersistentClient(path=PERSIST_DIRECTORY),
        ids=ids,
    )
    db.persist()
    db = None


def parse_split_text(documents: List[Document]) -> list:
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    return texts


DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    texts = parse_split_text(documents)
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE_TYPE},
    )
    # change the embedding type here if you are running into issues.
    # These are much smaller embeddings and will work for most appications
    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
    # main.py file.

    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    all_files = os.listdir(SOURCE_DIRECTORY)
    load_vector_db(
        texts, embeddings, ids=[uuid.uuid5(uuid.NAMESPACE_DNS, fn) for fn in all_files]
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
