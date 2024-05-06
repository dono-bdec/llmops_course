from sentence_transformers import SentenceTransformer
import numpy
from numpy.linalg import norm


# Let's go back to our original default embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Let's define our input sentences
input_sentences = ["There is a cat near my house", "My house is made of bricks"]

#Let's create vector_embeddings
vector_embeddings = embedding_model.encode(input_sentences)

# Search statement embedding
search_statement = "Cats often live on the streets"
search_embedding = embedding_model.encode(search_statement)

# define our distance metric
def calculate_cs(a, b):
    return numpy.dot(a, b)/(norm(a)*norm(b))

# Check the cosine similarity
print(f"Search: {search_statement}")
for search, input in zip(vector_embeddings, input_sentences):
    print(input, " -> similarity score = ",
         calculate_cs(search, search_embedding))