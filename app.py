# app.py

from flask import Flask, request, jsonify
from mongodb_search import retrieve_similar_documents
from openai_gpt import generate_rag_response
from data_processing import get_embedding

app = Flask(__name__)


# Chatbot route
@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get("query")

    # Step 1: Get the embedding of the user query
    query_embedding = get_embedding(user_query)

    # Step 2: Retrieve similar documents from MongoDB
    retrieved_docs = retrieve_similar_documents(query_embedding, top_n=5)

    # Step 3: Generate a response using OpenAI GPT-4
    response = generate_rag_response(retrieved_docs, user_query)

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)