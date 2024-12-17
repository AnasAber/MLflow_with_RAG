from llama_index.core.evaluation import EmbeddingQAFinetuneDataset, RetrieverEvaluator, generate_question_context_pairs
from llama_index.core import (Settings, SimpleDirectoryReader, VectorStoreIndex, QueryBundle)
from mlflow.models import convert_input_example_to_serving_input, validate_serving_input
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from chroma_db_functions import check_and_process_documents
from model import generate_response, generate_response_groq
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from main_reasoning import reasoning
import nest_asyncio
import pandas as pd
import warnings
import asyncio
import mlflow
import os


nest_asyncio.apply()
warnings.filterwarnings('ignore')


import logging
logger = logging.getLogger("mlflow")
logger.setLevel(logging.DEBUG)


class RAGTuningPipeline(mlflow.pyfunc.PythonModel):
    def __init__(self, dataset_dir, chunk_size, top_k, model_name, embedder_name, 
                 dataset_name, chunk_questions, retriever_type):
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.model_name = model_name
        self.embedder_name = embedder_name
        self.dataset_name = dataset_name
        self.chunk_questions = chunk_questions
        self.retriever_type = retriever_type

    def load_context(self, context):
        # Initialize LLM and embedding model
        print("load the context")
        llm = LlamaIndexOllama(model=self.model_name, modelfile="Modelfile", request_timeout=120.00)
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:latest",
            base_url="http://localhost:11434"
        )
        Settings.llm = llm
        Settings.embed_model = embed_model

    def predict(self, context=None, input_data=None):
    
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:latest",
            base_url="http://localhost:11434"
        )
        Settings.embed_model = embed_model
        
        llm = LlamaIndexOllama(model=self.model_name, modelfile="Modelfile")
        Settings.llm = llm

        # Ensure input_data is a dictionary with a 'query' key
        if isinstance(input_data, pd.DataFrame):
            query = input_data['query'].iloc[0]
        elif isinstance(input_data, dict):
            query = input_data.get('query', '')
        else:
            query = str(input_data)

        print(f"\nProcessing query: {query}")

        print("Retrieving relevant nodes...")
        context_text = reasoning(query)
        print(f"Retrieved nodes text: {context_text[:100]}")

        # Generate response
        print("Generating response...")
        response = generate_response_groq(context_text, query)

        print(f"response: {response}")
        return {"query": query, "responses": str(response)}


    async def evaluate(self):
        # Load or generate QA dataset
        if os.path.exists(self.dataset_name):
            qa_dataset = EmbeddingQAFinetuneDataset.from_json(self.dataset_name)
        else:
            print("data does not exist")
            documents = SimpleDirectoryReader(self.dataset_dir).load_data()
            node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=100)
            nodes = node_parser.get_nodes_from_documents(documents)

            for idx, node in enumerate(nodes):
                node.id_ = f"node_{idx}"

            qa_dataset = generate_question_context_pairs(
                nodes, llm=self.llm, num_questions_per_chunk=self.chunk_questions
            )
            qa_dataset.save_json(self.dataset_name)

        print("will begin to evaluate...🍀")

        # Perform evaluation
        retriever_evaluator = RetrieverEvaluator.from_metric_names(["mrr", "hit_rate"], retriever=vector_retriever)

        eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)

        metric_dicts = []
        # Calculate metrics
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)


        full_df = pd.DataFrame(metric_dicts)

        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()

        # Log metrics in MLflow
        with mlflow.start_run():
            mlflow.log_metric("Hit Rate", hit_rate)
            mlflow.log_metric("MRR", mrr)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("embedder_name", self.embedder_name)
            mlflow.log_param("top_k", self.top_k)
            mlflow.log_param("chunk_size", self.chunk_size)
            mlflow.log_param("chunk_questions", self.chunk_questions)
            mlflow.log_param("retriever_type", self.retriever_type)
            mlflow.log_artifact(self.dataset_name)
        
        print("Evaluation ended! Results:\n")
        print(f"Hit rate: {hit_rate}\n")
        print(f"MRR: {mrr}\n")

        return {"Hit Rate": hit_rate, "MRR": mrr}
    


# Save the model
if __name__ == "__main__":


    print("documents loaded")

    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text:latest",
        base_url="http://localhost:11434"
    )
    Settings.embed_model = embed_model

    vector_index = check_and_process_documents()

    print("documents added to the vector store")
        
    model_params = {
        "dataset_dir": "../data/raw",
        "chunk_size": 512,
        "top_k": 8,
        "model_name": "llama3.2:latest",
        "embedder_name": "nomic-embed-text:latest",
        "dataset_name": "pg_eval_dataset_index_BERT.json",
        "chunk_questions" : 2,
        "retriever_type" : "vector_retriever",
    }

    experiment_name = "rag_deployement"

    # mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text:latest",
        base_url="http://localhost:11434"
    )
    Settings.llm = LlamaIndexOllama(model=model_params["model_name"], modelfile="Modelfile")

    # Create a proper example input
    example_input = { "query" : "Does BERT has an Encoder architecture or an Enocder-Decoder one?" }
    
    # # Convert input example to serving input
    serving_input = convert_input_example_to_serving_input(example_input)

    with mlflow.start_run():
        # Log the RAG pipeline
        mlflow.pyfunc.log_model(
            artifact_path="rag_deployement",
            python_model=RAGTuningPipeline(**model_params),
            conda_env={
                "channels": ["defaults", "conda-forge"],
                "dependencies": [
                    "python=3.11.5",
                    "mlflow",
                    "torch",
                    "pandas",
                    "pip",
                    {
                        "pip": [
                            "llama-index",
                            "llama-index-core",
                            "llama-index-llms-ollama",
                            "llama-index-retrievers-bm25",
                            "llama-index-llms-huggingface",
                            "llama-index-embeddings-ollama",
                            "llama-index-vector-stores-chroma",
                            "llama-index-llms-huggingface-api",
                            "scikit-learn",
                            "groq",
                            "langchain",
                            "langchain-community",
                            "einops",
                            "sentence-transformers"
                        ]
                    },
                ]
            },
            input_example=serving_input,
        )

        mlflow.log_param("LLM", model_params["model_name"])
        mlflow.log_param("Embedding Model", model_params["embedder_name"])

        mlflow.log_param("Rertiever Used", model_params["retriever_type"])
        mlflow.log_param("top_k", model_params["top_k"])
        mlflow.log_param("chunk_size", model_params["chunk_size"])


    print("Model logged to MLflow")
 