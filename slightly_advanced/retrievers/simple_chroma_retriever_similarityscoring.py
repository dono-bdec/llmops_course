# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pprint

# Access the file and load it
loader = CSVLoader("../example_data/menu.csv")
pages = loader.load()

# This is the same default embedding that chromadb uses
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

store = Chroma(
    embedding_function=embeddings, 
    collection_name="csv", 
    persist_directory='db',
)

semantic_chunk_retriever = store.as_retriever(search_kwargs={
                                              "k": 5, "score_threshold": 0.1,
                                            #   "filter": {'row': 4}
                                              },
                                              search_type="similarity_score_threshold")

pprint.pprint(semantic_chunk_retriever.invoke("chicken"))
