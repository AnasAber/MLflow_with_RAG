from dotenv import load_dotenv
from groq import Groq
import requests
import os

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

load_dotenv()

api_key = os.getenv("GROQ_API")

client = Groq(api_key=api_key)

def generate_response_groq(context, query):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Answer the question of the user based on this context {context}\n\nQuestion: {query}"}],
        model="llama3-70b-8192",
        temperature=0.2,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
        )
    response = chat_completion.choices[0].message.content
    return response