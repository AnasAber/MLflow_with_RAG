
import requests

def generate_response(context, query, model_name):
    payload = {
        "model": model_name,
        "prompt": f"{context}\n\nQuestion: {query}",
        "stream": False
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json=payload, 
            timeout=300  # 5-minute timeout
        )
        response.raise_for_status()
        return response.json()['response']
    except requests.RequestException as e:
        print(f"Error generating response: {e}")
        return None

