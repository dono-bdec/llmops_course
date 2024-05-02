import chromadb
import pprint

chroma_client = chromadb.PersistentClient(path="llmops_s1_db")
collection = chroma_client.get_collection(name="html_file")

results =collection.query(query_texts=["adminstrators?"])

pprint.pprint(results)