# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

#Access the file and load it
loader = CSVLoader("example_data/menu.csv")
pages = loader.load()

#This is the same default embedding that chromadb uses
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

store = Chroma.from_documents(
    pages, 
    embeddings, # Unlike directly using chromadb, we need to define the embeddings with langchain
    collection_name="csv_file", 
    persist_directory='llmops_s1_db',
)



