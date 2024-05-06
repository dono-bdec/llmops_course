from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#What does the vector embedding for this sentence using the above embedding model look like?
vector_embedding = embedding_model.encode("Cats often live on the streets")

print(vector_embedding)