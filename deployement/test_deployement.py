from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import mlflow.pyfunc
import asyncio

# Set embedding model
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

# this one is for BERT llama3.2:latest, which is good: 8045796c453043458a20121dd43d2d2b
# this one is for Monopoly: 441b46b7a3f9436e8cf524644400b187

# Load model
model_uri = "runs:/8045796c453043458a20121dd43d2d2b/rag_deployement"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Test the evaluate function
async def test_evaluate(data):
    # Call the predict function with None to trigger evaluation
    return await loaded_model.predict(data)

# Run the test
if __name__ == "__main__":
    input_data = {"query": "Does BERT has an Encoder architecture or an Enocder-Decoder one?"}
    input_data = None
    # print(loaded_model.predict(input_data))
    evaluation_results = asyncio.run(test_evaluate(input_data))
    print("Evaluation Results:", evaluation_results)

