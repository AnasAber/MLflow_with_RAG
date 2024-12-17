from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
        
load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")


def get_embeddings(text=None):
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code": True})
    return embeddings