import chromadb
import pprint
chroma_client = chromadb.PersistentClient(path="menu_db")
collection = chroma_client.get_collection(name="menu_collection")

results = collection.get(where={"chef_recommends": "Y"}, include=["documents", "metadatas"])

print("Chef recommended items")
pprint.pprint(results)

results = collection.get(where={"diet_type": "NV"}, include=["documents", "metadatas"])

print("")
print("Diet Type NV items")
pprint.pprint(results)