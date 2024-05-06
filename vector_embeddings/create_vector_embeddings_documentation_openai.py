# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# The documentation has been downloaded from https://docs.readthedocs.io/en/stable/
loader = ReadTheDocsLoader('example_data/read_the_docs_documentation/')
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=len,
    separators=['\n\n', '\n', ' ', '']
)

all_metadata = []
all_text_chunks = []
source_url = "https://docs.readthedocs.io/en/stable/"

# We'll now go ever each doc's page content, split the text, store it in the chroma db
# Let's not forget to add the metadata link either
for i in range(0,len(docs)):
    curr_page_docs = text_splitter.split_text(docs[i].page_content)
    for eachPage in curr_page_docs:
        all_text_chunks.append(eachPage)
        all_metadata.append({"source_link": docs[i].metadata['source'].replace("example_data/read_the_docs_documentation/", "https://docs.readthedocs.io/en/stable/")})


# We can now load this into Chroma
# #This is the same default embedding that chromadb uses
embeddings = AzureOpenAIEmbeddings(
    model="ADA_RAG_DONO_DEMO",
    api_key=pinginstructorforkey,
    api_version="2024-02-01",
    azure_endpoint=pinginstructorforurl
)

embeddings2 = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

start_time_openai = time.time()


store = Chroma.from_texts(
    all_text_chunks, 
    embeddings, # Unlike directly using chromadb, we need to define the embeddings with langchain
    collection_name="readthedocs_openai", 
    persist_directory='llmops_s3_db',
    metadatas=all_metadata
)

end_time_openai = time.time()

start_time_default = time.time()
store2 = Chroma.from_texts(
    all_text_chunks, 
    embeddings2, # Unlike directly using chromadb, we need to define the embeddings with langchain
    collection_name="readthedocs_default", 
    persist_directory='llmops_s3_db',
    metadatas=all_metadata
)
end_time_default = time.time()



print(f"Time taken for Open AI: {(end_time_openai-start_time_openai)}")
print(f"Time taken for Default: {(end_time_default-start_time_default)}")

