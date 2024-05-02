from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

#Access the file and load it
search_term = "2024 Indian Premier League"
pages = WikipediaLoader(query=search_term, load_max_docs=1).load()


#This is the same default embedding that chromadb uses
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

store = Chroma.from_documents(
    pages, 
    embeddings, # Unlike directly using chromadb, we need to define the embeddings with langchain
    collection_name="wikipedia", 
    persist_directory='llmops_s1_db',
)



