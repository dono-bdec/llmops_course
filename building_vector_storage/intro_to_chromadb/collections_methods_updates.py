# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
import pprint
chroma_client = chromadb.PersistentClient(path="llmops_s1_db")

collection = chroma_client.get_collection(name="my_collection")

collection.update(ids=["id1"], documents=["This is a query document"])

results = collection.query(
    query_texts=["This is a query document"],
    n_results=2
)

pprint.pprint(results)
