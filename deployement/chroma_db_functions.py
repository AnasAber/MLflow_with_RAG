import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from data.process_data import load_documents, embed_and_store_documents, split_documents
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_community.vectorstores import Chroma
from  get_embeddings import get_embeddings


CHROMA_PATH = "..data/chroma"


# load the data
def get_chroma_db(get_embeddings=get_embeddings()):
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings)


def retrieve_documents(query, top_k=5):
    chroma_db = get_chroma_db()
    print("#"*100 + "\n\n")

    print("Retrieving documents...")
    results = chroma_db.similarity_search_with_score(query, top_k)
    context_text= "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    print("Documents before reranking: ", context_text)

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

