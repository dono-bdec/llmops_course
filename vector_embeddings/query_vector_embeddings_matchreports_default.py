# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
import pprint


chroma_client = chromadb.PersistentClient(path="llmops_s3_db")
collection = chroma_client.get_collection(name="alltextfiles_default")

results=collection.query(query_texts=["Emirates"], n_results=3)

pprint.pprint(results)
