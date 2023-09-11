# Multi-document Analyzer - Chat with a bot about your files in private

## Overview
Chat about your docs in private
Main techonology stack: LLM, HuggingFace, LangChain and Streamlit 

Features:
1. Upload documents in PDF,CSV,TXT,Excel formats to your knowledge store
2. Upload transcripts from YT videos to your knowledge store
3. Uses LLMs to comprehend, summarize, create, and anticipate new material

## Hardware Requirements
1. Best scenario is to have GPUs, but it also works on CPU (with the expected delay in response)
2. 16 GB of vRAM or more


## Dependencies install 

```bash
poetry install
```

## Run the application
In the root folder add a .env file with the proper HUGGINGFACE_API_KEY and HUGGINGFACEHUB_API_TOKEN variables

```bash
poetry run streamlit run main.py
```

A browser instance will be opened to talk to chatbot about your Documents

## Application Usage
Write different questions about USA constitution