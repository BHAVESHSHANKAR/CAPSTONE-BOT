import os
from flask import Flask, request, jsonify
import requests
import dotenv
# Initialize the Flask application
app = Flask(__name__)
dotenv.load_dotenv()  # Load environment variables from .env file
# --- Configuration ---
# Get configuration from environment variables
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable is not set")

API_URL = os.getenv('NVIDIA_API_URL', "https://integrate.api.nvidia.com/v1/chat/completions")

# --- System Prompt to Define the Agent's Behavior ---
# This instruction tells the AI what its job is and what topics it's limited to.
SYSTEM_PROMPT = """
You are a specialized AI assistant. Your sole purpose is to answer questions about 
blockchain, security, cybersecurity, and technology. If a user asks about any 
other topic (like cooking, history, sports, etc.), you MUST politely refuse to 
answer. State that you can only provide information on your specialized fields. 
Do not answer questions outside of your scope.
"""
@app.route('/')
def index():
    return "Welcome to the NVIDIA AI Chatbot! Use the /ask endpoint to ask questions."
# --- API Endpoint for Chatting ---
@app.route('/ask', methods=['POST'])
def ask_agent():
    """
    This endpoint receives a question and returns an answer from the specialized AI.
    JSON payload should be: {"question": "Your question here"}
    """
    # Check for a valid request body
    request_data = request.get_json()
    if not request_data or 'question' not in request_data:
        return jsonify({"error": "Request body must be JSON with a 'question' key."}), 400

    user_question = request_data['question']

    # Prepare the headers and payload for the NVIDIA API
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "meta/llama3-8b-instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ],
        "temperature": 0.5,
        "top_p": 1.0,
        "max_tokens": 1024,
        "stream": False
    }

    # Call the NVIDIA API
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes

        api_response = response.json()
        agent_answer = api_response['choices'][0]['message']['content']

        return jsonify({"answer": agent_answer})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to communicate with NVIDIA API: {e}"}), 502
    except (KeyError, IndexError) as e:
        return jsonify({"error": "Invalid response format from NVIDIA API.", "details": str(e)}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)