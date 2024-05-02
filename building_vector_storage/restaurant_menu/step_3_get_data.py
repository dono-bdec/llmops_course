# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

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