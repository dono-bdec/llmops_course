import chromadb
import pprint


chroma_client = chromadb.PersistentClient(path="llmops_s1_db")
collection = chroma_client.get_collection(name="csv_file")

results =collection.query(query_texts=["Mutton"])

pprint.pprint(results)