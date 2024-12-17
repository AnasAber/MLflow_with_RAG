# from llama_index.core.evaluation import EmbeddingQAFinetuneDataset, RetrieverEvaluator, generate_question_context_pairs
# from llama_index.core import (Settings, SimpleDirectoryReader, VectorStoreIndex, QueryBundle)
# from mlflow.models import convert_input_example_to_serving_input, validate_serving_input
# from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
# from llama_index.llms.ollama import Ollama as LlamaIndexOllama
# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.retrievers.bm25 import BM25Retriever
# from model import generate_response
# import nest_asyncio
# import pandas as pd
# import warnings
# import asyncio
# import mlflow
# import json
# import os

# llm = LlamaIndexOllama(model="llama3.2:1b", modelfile="Modelfile")

# # Set embedding model
# Settings.embed_model = OllamaEmbedding(
#     model_name="nomic-embed-text:latest",
#     base_url="http://localhost:11434"
# )


# async def testing():
#     documents = SimpleDirectoryReader("../data").load_data()
#     node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)
#     nodes = node_parser.get_nodes_from_documents(documents)

#     for idx, node in enumerate(nodes):
#         node.id_ = f"node_{idx}"

#     vector_index = VectorStoreIndex(nodes)
#     vector_retriever = vector_index.as_retriever(similarity_top_k=8, similarity_cutoff=0.5)

#     qa_dataset = EmbeddingQAFinetuneDataset.from_json("pg_eval_dataset_index.json")

#     print("will begin to evaluate...🍀")

#     # print(f"here's what's in qa_dataset: {qa_dataset}")
#     # Perform evaluation
#     retriever_evaluator = RetrieverEvaluator.from_metric_names(["mrr", "hit_rate"], retriever=vector_retriever)
#     print(f"retrieval evaluator: {retriever_evaluator}")

#     eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
#     print(f"evaluator results : {eval_results}")

#     metric_dicts = []
#     # Calculate metrics
#     for eval_result in eval_results:
#         metric_dict = eval_result.metric_vals_dict
#         metric_dicts.append(metric_dict)
#     print(f"metric dictionnaries: {metric_dict}")
        
#     full_df = pd.DataFrame(metric_dicts)

#     print(f"Hit rate: {full_df['hit_rate']}")
#     print(f"MRR: {full_df['mrr']}")

#     hit_rate = full_df["hit_rate"].mean()
#     mrr = full_df["mrr"].mean()

#     print(f"Hit rate: {hit_rate}")
#     print(f"MRR: {mrr}")

# asyncio.run(testing())
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import requests

# Set embedding model
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

url = "http://127.0.0.1:5001/invocations"
data = {"instances": [{"query": "Tell me about BERT architecture"}]}
response = requests.post(url, json=data)
print(response.json())
