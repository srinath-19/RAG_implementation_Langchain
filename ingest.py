import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

MODEL = "gpt-4.1-nano"

# ✅ Use the project folder (where ingest.py lives) as the base
PROJECT_ROOT = Path(__file__).parent
DB_NAME = str(PROJECT_ROOT / "vector_db")
KNOWLEDGE_BASE = str(PROJECT_ROOT / "knowledge-base")

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    if not os.path.isdir(KNOWLEDGE_BASE):
        raise FileNotFoundError(
            f"KNOWLEDGE_BASE not found: {KNOWLEDGE_BASE}\n"
            f"Create it and add .md files, or fix the path."
        )

    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []

    for folder in folders:
        if not os.path.isdir(folder):
            continue

        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()

        for doc in folder_docs:
            # Skip empty docs early
            if not doc.page_content or not doc.page_content.strip():
                continue
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    print(f"Loaded {len(documents)} documents from: {KNOWLEDGE_BASE}")
    return documents


def create_chunks(documents):
    if not documents:
        raise ValueError("No documents loaded. Nothing to chunk.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = [c for c in text_splitter.split_documents(documents) if c.page_content.strip()]

    print(f"Created {len(chunks)} chunks")
    return chunks


def create_embeddings(chunks):
    if not chunks:
        raise ValueError("No chunks to embed. Check your knowledge-base content and loader glob.")

    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )

    collection = vectorstore._collection
    count = collection.count()

    got = collection.get(limit=1, include=["embeddings"])
    sample_embedding = got["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")