# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
import pprint
chroma_client = chromadb.PersistentClient(path="menu_db")
collection = chroma_client.get_collection(name="menu_collection")


results =collection.query(query_texts=["mutton"])

pprint.pprint(results)