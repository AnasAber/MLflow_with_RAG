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