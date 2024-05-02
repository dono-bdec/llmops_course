import chromadb
import pprint


chroma_client = chromadb.PersistentClient(path="llmops_s1_db")
collection = chroma_client.get_collection(name="wikipedia")

results =collection.query(query_texts=["Format of 2024 IPL"])

pprint.pprint(results)