from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pprint


alice_in_wonderland = ""

with open("../example_data/alice.txt") as f:
  alice_in_wonderland = f.read()

embeddings = AzureOpenAIEmbeddings(
    model="ADA_RAG_DONO_DEMO",
    api_key="api_key",
    api_version="2024-02-01",
    azure_endpoint="azure_endpoint"
)



semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")


semantic_chunks = semantic_chunker.create_documents([alice_in_wonderland])

semantic_chunk_vectorstore = Chroma.from_documents(
    semantic_chunks, 
    embedding=embeddings, 
    collection_metadata={"hnsw:space": "cosine"}, #Change the distance function based on type of data
    persist_directory="db",
    collection_name="alice_in_wonderland_semantic"
    )



semantic_chunk_retriever = semantic_chunk_vectorstore.as_retriever(search_kwargs={"k" : 1})



pprint.pprint(semantic_chunk_retriever.invoke("Who has a pocket watch?"))