import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from chroma_db_functions import get_relevant_data, check_and_process_documents
from model import generate_response_groq


def format_context(context):
    return "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context)])


def reasoning(query):

    check_and_process_documents()

    print("#"*100 + "\n\n")
    
    results = get_relevant_data(query[0])

    response = generate_response_groq(results, query)
    return response