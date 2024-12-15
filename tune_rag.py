from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    generate_question_context_pairs
)
import nest_asyncio
import pandas as pd
import warnings
import argparse
import asyncio
import mlflow
import os

nest_asyncio.apply()
warnings.filterwarnings('ignore')



llm = LlamaIndexOllama(model="mistral", modelfile="Modelfile")
embed_model = OllamaEmbedding(
                    model_name="nomic-embed-text:latest",
                    base_url="http://localhost:11434"
                )

Settings.llm = llm
Settings.embed_model = embed_model


def parse_args():
    parser = argparse.ArgumentParser(description="Tune RAG Model MLflow")
    parser.add_argument('--dataset_dir', type=str, default="./data", help="Directory for the dataset")
    parser.add_argument('--chunk_size', type=int, default=512, help="Chunk size for splitting documents")
    parser.add_argument('--top_k', type=int, default=8, help="Top K similar nodes to retrieve")
    parser.add_argument('--model_name', type=str, default='mistral', help="Model name")
    parser.add_argument('--embedder_name', type=str, default='nomic-embed-text:latest', help="Embedder name")
    parser.add_argument('--dataset_name', type=str, default='pg_eval_dataset_index.json', help="Dataset name")
    parser.add_argument('--chunk_questions', type=int, default=2, help="Number of questions per chunk")
    parser.add_argument('--retriever_type', type=str, default="bm25_retriever", help="The retriever type")
    return parser.parse_args()



async def tune_rag(args):
    documents = SimpleDirectoryReader(args.dataset_dir).load_data()
    node_parser = SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)

    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"


    vector_index = VectorStoreIndex(nodes)


    # Multiple retrieval strategies
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes)
    vector_retriever = vector_index.as_retriever(similarity_top_k=args.top_k, similarity_cutoff=0.6)


    # Check if the dataset JSON file exists
    if os.path.exists(args.dataset_name):   
        # Load existing dataset
        qa_dataset = EmbeddingQAFinetuneDataset.from_json(args.dataset_name)
        print(qa_dataset)
        print(f"Loaded existing dataset from {args.dataset_name}")
    else:
        # Generate new dataset
        qa_dataset = generate_question_context_pairs(
            nodes, 
            llm=llm, 
            num_questions_per_chunk=args.chunk_questions
        )
        print(f"this is the qa_dataset outside the function: {qa_dataset}")
        
        # Save the newly generated dataset
        qa_dataset.save_json(args.dataset_name)

        qa_dataset = EmbeddingQAFinetuneDataset.from_json(qa_dataset)

        print(f"Generated and saved new dataset to {args.dataset_name}")

    metrics = ["mrr", "hit_rate"]
    retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=vector_retriever)
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    with mlflow.start_run():
        mlflow.log_metric("Hit Rate", hit_rate)
        mlflow.log_metric("MRR", mrr)

        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("embedder_name", args.embedder_name)
        mlflow.log_param("top_k", args.top_k)
        mlflow.log_param("chunk_size", args.chunk_size)
        mlflow.log_param("chunk_questions", args.chunk_questions)
        mlflow.log_param("retriever type", args.retriever_type)

        mlflow.log_artifact(args.dataset_name)

    print("Run has been completed. View your results at http://127.0.0.1:5000")

if __name__ == "__main__":
    args = parse_args()
    mlflow.set_experiment("rag_tuning")
    asyncio.run(tune_rag(args))
