from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Access the file and load it
loader = UnstructuredHTMLLoader("example_data/2024iplhtml.html")
pages = loader.load()

#This is the same default embedding that chromadb uses
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

store = Chroma.from_documents(
    pages, 
    embeddings, # Unlike directly using chromadb, we need to define the embeddings with langchain
    collection_name="html_file", 
    persist_directory='llmops_s1_db',
)



