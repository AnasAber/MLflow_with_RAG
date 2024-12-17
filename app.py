import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)
from deployement.chroma_db_functions import close_chroma_db_connection, check_and_process_documents
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from deployement.main_reasoning import reasoning
from flask_cors import CORS
import requests
import shutil
import time
import json

app = Flask(__name__)
CORS(app)  # This allows all origins

# Define the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'raw')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# PROMPT_TEMPLATE = """
# Answer this question in a clear, unboring matter, based on the following context:
# {context}
# -----
# Answer this question based on the above context, without citing the context in your answer:
# {question};/
# Answer:
# """

def stream_response(response_text):
    """Stream the response one character at a time to simulate typing."""
    delay = 0.0001  # Adjust this delay to control the typing speed
    for char in response_text:
        yield char
        time.sleep(delay)


MLFLOW_MODEL_URL = "http://127.0.0.1:5001/invocations"


# @app.route('/query', methods=['POST'])
# def handle_query():
#     data = request.json
#     print(f"here's the data request: {data}")
#     query = data.get('query')
#     if not query:
#         return jsonify({"error": "No query provided"}), 400
    
#     try:
#         # Prepare data for MLflow model
#         print("i'm before mlflow_payload")
#         mlflow_payload = {"instances": [{"query": query}]}
        
#         # Send request to MLflow-served model
#         print("i'm before response")
#         response = requests.post(MLFLOW_MODEL_URL, json=mlflow_payload)
#         print("i'm after mlflow_payload")
        
#         # Check if the request was successful
#         if response.status_code == 200:
#             # Extract the response from the MLflow model
#             print("If this shows up, it means this should work")

#             model_response = response.json()

#             print(f"this is the model's response: {model_response}")

#             response_text = model_response.get('predictions', {}).get('responses', 'No response')
            
#             print(f"this is the response_Text : {response_text}")
#             return jsonify({"response": response_text})
#         else:
#             print("i'm in error that means response is not working")
#             return jsonify({"error": f"MLflow model request failed: {response.text}"}), 500
    
#     except requests.RequestException as e:
#         return jsonify({"error": f"Request to MLflow model failed: {str(e)}"}), 500


@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    print(f"Received data: {data}")
    
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Modify payload to match exact MLflow serving expectations
        mlflow_payload = {
            "dataframe_split": {
                "columns": ["query"],
                "data": [[query]]
            }
        }
        
        print(f"Payload to MLflow: {json.dumps(mlflow_payload)}")
        
        # Send request to MLflow model
        response = requests.post(
            MLFLOW_MODEL_URL, 
            json=mlflow_payload,
        )
        
        print(f"MLflow Response Status: {response.status_code}")
        print(f"MLflow Response Content: {response.text}")
        
        # Check if the request was successful
        if response.status_code == 200:
            model_response = response.json()
            print(f"Parsed Model Response: {model_response}")
            
            # Extract response from the nested structure
            response_text = model_response.get('predictions', {}).get('responses', 'No response found')
            
            print(f"Extracted Response Text: {response_text}")
            return jsonify({"response": response_text})
        else:
            return jsonify({
                "error": f"MLflow model request failed", 
                "status_code": response.status_code,
                "details": response.text
            }), 500
    
    except requests.RequestException as e:
        print(f"Request Exception: {e}")
        return jsonify({"error": f"Request to MLflow model failed: {str(e)}"}), 500
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Here you would typically process the file and add it to Chroma DB
        # You can call your processing function here
        check_and_process_documents()
        return jsonify({"message": f"File {filename} uploaded successfully to {file_path}"}), 200
    


# Route to clear doc data
@app.route('/clear_cv_data', methods=['POST'])
def clear_cv_data():
    chroma_folder = './data/chroma'

    try:

        close_chroma_db_connection()

        # Check if the chroma folder exists and delete its contents
        if os.path.exists(chroma_folder):
            for filename in os.listdir(chroma_folder):
                file_path = os.path.join(chroma_folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    # shutil.rmtree(file_path)
                    shutil.rmtree(chroma_folder)
        return jsonify({"message": "ChromaDB data cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)