from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from deployement.get_embeddings import get_embeddings
from concurrent.futures import ThreadPoolExecutor



CHROMA_PATH = "./data/chroma"
DATA_PATH = "./data/raw/"


def load_documents():
    document_loader = PyPDFDirectoryLoader("./data/")
    print("Loading documents...")
    return document_loader.load()


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    docs = []
    print("Splitting documents...")
    for document in documents:
        for chunk in text_splitter.split_text(document.page_content):
            docs.append(Document(page_content=chunk, metadata={"source": document.metadata["source"]}))
    print("Documents split successfully.")
    return docs


def embed_and_store_documents(chunks):
    chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    
    # Use ThreadPoolExecutor to process chunks into Document objects
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on CPU cores
        documents = list(executor.map(lambda chunk: Document(page_content=chunk.page_content), chunks))
    
    # Add the Document objects to ChromaDB
    chroma_db.add_documents(documents)
    chroma_db.persist()

    print("Documents embedded and stored successfully.")