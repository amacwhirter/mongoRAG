# mongodb_search.py

from pymongo import MongoClient
from config import mongo_uri, database_name, collection_name


# Function to retrieve similar documents from MongoDB
def retrieve_similar_documents(query_embedding, top_n=5):
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]

    # Print query embedding to debug
    # print(f"Query embedding: {query_embedding}")

    # Perform a vector similarity search using the 'embedding' field
    try:
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 500,
                    "limit": 10
                }
            }
        ])

        # Print the raw results for debugging
        docs = list(results)
        print(f"Retrieved {len(docs)} documents.")
        # print(f"Retrieved {docs} documents.")

        if len(docs) == 0:
            print("No matching documents found.")

        return docs

    except Exception as e:
        print(f"Error in MongoDB query: {e}")
        return []
