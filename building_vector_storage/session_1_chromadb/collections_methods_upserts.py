import chromadb
import pprint

chroma_client = chromadb.PersistentClient(path="llmops_s1_db")

collection = chroma_client.get_collection(name="my_collection")

collection.upsert(ids=["id1","id3"], documents=["This is a new query document","This is also another document"])

results = collection.query(
    query_texts=["This is a query document"],
    n_results=3
)

pprint.pprint(results)

