# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
import pprint

chroma_client = chromadb.PersistentClient(path="llmops_s1_db")

collection = chroma_client.get_collection(name="my_collection")

results = collection.get(ids=["id1"])

pprint.pprint(results)