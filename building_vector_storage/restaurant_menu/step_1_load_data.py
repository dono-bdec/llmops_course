# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import csv
import chromadb

# Read the data
with open('menu.csv') as file:
    menu_lines = csv.reader(file)
    menu_documents = []
    menu_metadatas = []
    menu_ids = []
    
    for i, menu_line in enumerate(menu_lines):
        if i==0:
            # Skip the first row (the column headers)
            continue

        menu_documents.append(menu_line[1])
        menu_metadatas.append({"diet_type": menu_line[2], "chef_recommends": menu_line[3]})
        menu_ids.append(menu_line[0])

# Load it into ChromaDB
chroma_client = chromadb.PersistentClient(path="menu_db")
collection = chroma_client.create_collection(name="menu_collection")

collection.add(
    documents=menu_documents,
    metadatas=menu_metadatas,
    ids=menu_ids
)

print(collection.count())



