### Project ongoing...🥀

This project is for people that want to use MLflow to evaluate their RAG pipeline.

The project uses:
- `LlamaIndex` as an orchestrator
- `Ollama` and `HuggingfaceLLMs`
- `MLflow` as an MLOps framework

![Project Overview Diagram](images/mlflow_overview.png)
### How to start

1. Clone the repository
```bash
git clone https://github.com/AnasAber/RAG_in_CPU.git
```

2. Install the dependencies
```bash
pip install -r requirements.txt
```

3. Notebook Prep:
- Put your own data files in the data/ folder
- Go to the notebook, and replace "api_key_here" with your huggingface_api_key
- If you have GPU, you're fine, if not, run it on google colab, and make sure to download the json file output at the end of the run.

4. Open two terminals:
```bash
python tune_rag.py
```
And after the run, do:
```bash
mlflow ui
```

