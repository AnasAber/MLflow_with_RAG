import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from data.process_data import load_documents, embed_and_store_documents, split_documents
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_community.vectorstores import Chroma
from deployement.get_embeddings import get_embeddings
import sqlite3

CHROMA_PATH = "..data/chroma"

# load the data
def get_chroma_db(get_embeddings=get_embeddings(), in_memory=True):
    if in_memory:
        # In-memory ChromaDB (much faster)
        return Chroma(embedding_function=get_embeddings)
    else:
        # Disk-based ChromaDB
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings)


def retrieve_documents(query, top_k=5):
    chroma_db = get_chroma_db()
    print("#"*100 + "\n\n")

    print("Retrieving documents...")
    results = chroma_db.similarity_search_with_score(query, top_k)
    context_text= "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    print("Documents: ", context_text)

    return context_text

def format_context(context):
    return "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context)])


def get_relevant_data(query):
    retrieved_chunks = retrieve_documents(query)
    # reranked_chunks = reranked_documents(query, retrieved_chunks)
    return retrieved_chunks


def add_to_chroma_db(reranked_chunks):
    chroma_db = get_chroma_db()
    chroma_db.add_documents(reranked_chunks)
    chroma_db.persist()


def check_and_process_documents():
    path = "../data/chroma"
    print(f"Checking if path exists: {path}")
    
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        
        documents = load_documents()
        print("Documents loaded")
        
        chunks = split_documents(documents)
        print("Documents split into chunks")
        
        embed_and_store_documents(chunks)
        print("Documents embedded and stored")
    else:
        print(f"Path already exists: {path}")

def close_chroma_db_connection():
    try:
        chroma_db_connection = get_chroma_db()
        if chroma_db_connection is not None:
            chroma_db_connection.delete_collection()
            chroma_db_connection.persist()
            chroma_db_connection = None

        # Forcefully close SQLite connection
        conn = sqlite3.connect('../data/chroma/chroma.sqlite3')
        conn.close()
        print("ChromaDB connection closed successfully.")
    except Exception as e:
        print(f"Error closing ChromaDB connection: {e}")
