import ollama
from sklearn.metrics.pairwise import cosine_similarity


def get_embedding(text: str):
    response = ollama.embeddings(
        model="nomic-embed-text:latest",
        prompt=text
    )
    return response['embedding']


docs = [
    "The Eiffel Tower in Paris is a wrought-iron structure known for its iconic design and panoramic city views.",
    "The Great Wall of China is an ancient series of fortifications built to protect against invasions and spans thousands of miles.",
    "The Statue of Liberty in New York symbolizes freedom and democracy, and was a gift from France to the United States."
]


query = "What is the iconic structure in Paris known for its panoramic views?"

doc_embeddings = [get_embedding(doc) for doc in docs]
query_embedding = get_embedding(query)

score = cosine_similarity([query_embedding], doc_embeddings)[0]

print("Similarity Scores:", score)

# Get the index of the document with the highest similarity score
best_index, best_scrore = max(enumerate(score), key=lambda x: x[1])

print(
    f"Best Match: {docs[best_index]} with a similarity score of {best_scrore:.4f}")
