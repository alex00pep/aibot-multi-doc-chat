import os
import time
from uuid import uuid4
import uuid
import torch
import chromadb
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    SOURCE_DIRECTORY,
)
from langchain.embeddings import (
    HuggingFaceInstructEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from ingest import (
    files_difference,
    load_documents,
    load_single_document,
    load_vector_db,
    parse_split_text,
)
from model_loader import load_model, load_remote_hosted_model

load_dotenv()


def model_memory():
    # Adding history to the model.
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"], template=template
    )
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory


# Sidebar contents
with st.sidebar:
    st.title("🤗💬 Converse with your Data")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [LocalGPT](https://github.com/PromtEngineer/localGPT)
 
    """
    )
    add_vertical_space(5)
    st.write("Made with ❤️ by [AlexPepito](https://github.com/alex00pep)")


DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


if "result" not in st.session_state:
    # Run the document ingestion process.
    run_langest_commands = ["python", "ingest.py"]
    run_langest_commands.append("--device_type")
    run_langest_commands.append(DEVICE_TYPE)

    result = subprocess.run(run_langest_commands, capture_output=True)
    st.session_state.result = result

# Define the retreiver
# load the vectorstore
if "EMBEDDINGS" not in st.session_state:
    EMBEDDINGS = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE}
    )
    # EMBEDDINGS = HuggingFaceInstructEmbeddings(
    #     model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE}
    # )
    st.session_state.EMBEDDINGS = EMBEDDINGS

if "DB" not in st.session_state:
    DB = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=st.session_state.EMBEDDINGS,
        client=chromadb.PersistentClient(path=PERSIST_DIRECTORY),
    )
    st.session_state.DB = DB

if "RETRIEVER" not in st.session_state:
    RETRIEVER = DB.as_retriever()
    st.session_state.RETRIEVER = RETRIEVER

if "LLM" not in st.session_state:
    LLM = load_model(
        device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME
    )
    # LLM = load_remote_hosted_model(repo_id=MODEL_ID)
    st.session_state["LLM"] = LLM


if "QA" not in st.session_state:
    prompt, memory = model_memory()

    QA = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=RETRIEVER,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

    st.session_state["QA"] = QA

st.header("Multi-document Analyzer 💬")
new_files = st.file_uploader(
    "Upload your Word/PDF/txt/Excel/Markdown files",
    accept_multiple_files=True,
    type=[name.replace("\.\s", "") for name in DOCUMENT_MAP.keys()],
)
st.subheader("Your documents")

if st.button("Ingest/Process"):
    if new_files:
        # Split and Load files to Vector DB
        with st.spinner("Processing files..."):
            filenames = []
            for tmp_file in new_files:
                fn = os.path.join(SOURCE_DIRECTORY, os.path.basename(tmp_file.name))
                filenames.append(fn)
                with open(fn, "wb") as f:
                    f.write(tmp_file.getbuffer())
            st.success("Saved Files")

            # new_files = files_difference(SOURCE_DIRECTORY, new_files=filenames)
            # texts = parse_split_text([load_single_document(fn) for fn in new_files])

            # load_vector_db(
            #     texts,
            #     st.session_state.EMBEDDINGS,
            #     ids=[uuid.uuid5(uuid.NAMESPACE_DNS, fn) for fn in new_files],
            # )

            # load_db(documents=all_files, embeddings=st.session_state.EMBEDDINGS)
    else:
        st.warning("Files not selected")

st.markdown("---")

# Create a text input box for the user
prompt = st.text_input("Input your prompt here")
# while True:

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    with st.spinner("Generating response..."):
        start = time.monotonic()
        response = st.session_state["QA"]({"query": prompt})
        answer, docs = response["result"], response["source_documents"]
        end = time.monotonic()
        st.write(f"Answered in {round((end-start)/60, 2)} minutes")
        # ...and write it out to the screen
        st.write(answer)

        # With a streamlit expander
        with st.expander("Source Documents"):
            # Find the relevant pages
            search = st.session_state.DB.similarity_search_with_score(prompt)
            # Write out the first
            for i, doc in enumerate(search):
                # print(doc)
                st.write(
                    f"Source Document # {i+1} : {doc[0].metadata['source'].split('/')[-1]}"
                )
                st.write(doc[0].page_content)
                st.write("--------------------------------")
else:
    st.warning("Please provide a prompt")
