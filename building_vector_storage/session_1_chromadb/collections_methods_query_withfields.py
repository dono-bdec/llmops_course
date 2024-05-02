import chromadb

chroma_client = chromadb.PersistentClient(path="llmops_s1_db")

collection = chroma_client.create_collection(name="my_collection")


collection.add(
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    include=["metadata", "documents","distances"]
)

print(results)
