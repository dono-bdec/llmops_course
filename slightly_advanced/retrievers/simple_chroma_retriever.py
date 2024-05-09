# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import csv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pprint

#Access the file and load it - we could have just used the CSV Loader but we cannot capture metadata easily in that case
with open('../example_data/menu.csv') as file:
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

#This is the same default embedding that chromadb uses
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

store = Chroma.from_texts(
    menu_documents, 
    embeddings, # Unlike directly using chromadb, we need to define the embeddings with langchain
    collection_name="csv", 
    persist_directory='db',
    metadatas=menu_metadatas,
    # collection_metadata={"hnsw:space": "cosine"} #Change the distance function based on type of data
)

semantic_chunk_retriever = store.as_retriever()

pprint.pprint(semantic_chunk_retriever.invoke("Chicken"))



