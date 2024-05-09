from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


example_text=""

with open('../example_data/wc.txt', 'r') as file:
    example_text = file.read()

# Split your website into big chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
chunks = text_splitter.split_text(example_text)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma.from_texts(texts=chunks, embedding=embeddings)

similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

print("Vanilla docs!")
print(similarity_retriever.invoke("wickets"))

print("MMR docs!")
print(mmr_retriever.invoke("wickets"))


