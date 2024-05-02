import chromadb
import pprint
chroma_client = chromadb.PersistentClient(path="menu_db")
collection = chroma_client.get_collection(name="menu_collection")


results =collection.query(query_texts=["mutton"])

pprint.pprint(results)