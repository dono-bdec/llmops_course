# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
import pprint


chroma_client = chromadb.PersistentClient(path="llmops_s2_db")
collection = chroma_client.get_collection(name="readthedocs")

results=collection.query(query_texts=["internationalizing", "logs for security"], n_results=1)

print("---------------------")
print("Query Text: internationalizing")
pprint.pprint(results["metadatas"][0][0])
pprint.pprint(results["documents"][0][0])
print("---------------------")
print("Query Text: Security logs")
pprint.pprint(results["metadatas"][1][0])
pprint.pprint(results["documents"][1][0])
print("---------------------")
