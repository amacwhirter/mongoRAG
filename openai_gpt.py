# openai_gpt.py

from openai import OpenAI
from config import openai_api_key

client = OpenAI(
    api_key=openai_api_key,  # this is also the default, it can be omitted
)


# Function to generate a response using GPT-4 (chat model)
def generate_rag_response(retrieved_docs, user_query):
    # Combine the text from the retrieved documents to use as context
    context = "\n".join([doc['text_column'] for doc in retrieved_docs])

    # Prompt to OpenAI GPT-4
    prompt = f"Based on the following information:\n{context}\n\nAnswer the following question: {user_query}"

    # Updated API call using `openai.ChatCompletion.create()` for chat models like GPT-4
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )

    # Return the generated response
    return response.choices[0].message.content.strip()