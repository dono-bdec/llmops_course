import chromadb
import pprint


chroma_client = chromadb.PersistentClient(path="llmops_s1_db")
collection = chroma_client.get_collection(name="pdf_file")

results =collection.query(query_texts=["What happens in fellowship of the ring?"])

pprint.pprint(results)