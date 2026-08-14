import os
import zipfile

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


ZIP_PATH = "data/raw.zip"
VECTORSTORE_PATH = "vectorstore"


print("Starting DRDO document ingestion...")


# --------------------------------
# 1. Read documents from ZIP
# --------------------------------

documents = []

with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:

    for file_name in zip_file.namelist():

        if file_name.lower().endswith(".txt"):

            text = zip_file.read(file_name).decode(
                "utf-8",
                errors="ignore"
            )

            text = text.strip()

            if text:

                document = Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "category": "Defence",
                        "dataset": "DRDO"
                    }
                )

                documents.append(document)


print(f"Documents loaded: {len(documents)}")


# --------------------------------
# 2. Split documents into chunks
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# --------------------------------
# 3. Create embeddings
# --------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 4. Create Chroma vector database
# --------------------------------

print("Creating Chroma vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTORSTORE_PATH
)


print("================================")
print("DRDO RAG PIPELINE COMPLETED")
print("================================")

print(f"Documents : {len(documents)}")
print(f"Chunks    : {len(chunks)}")
print("Vector DB : vectorstore/")