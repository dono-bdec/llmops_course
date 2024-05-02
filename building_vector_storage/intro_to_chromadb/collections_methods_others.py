# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb


chroma_client = chromadb.PersistentClient(path="llmops_s1")


collection = chroma_client.get_collection("my_collection")

print(collection.count())


pprint.pprint(collection.peek())