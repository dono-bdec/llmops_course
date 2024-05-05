from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings



# The documentation has been downloaded from https://docs.readthedocs.io/en/stable/
loader = ReadTheDocsLoader('example_data/read_the_docs_documentation/')
docs = loader.load()


# Check out the entire document loader's documentation here: 
# https://python.langchain.com/docs/integrations/document_loaders/readthedocs_documentation/

# Check what the page content looks like
# print(docs[0].page_content)
# print(docs[5].page_content)

# We can use the source link for metadata
# print(docs[12].metadata['source'].replace("example_data/read_the_docs_documentation/", "https://docs.readthedocs.io/en/stable/"))


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
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
store = Chroma.from_texts(
    all_text_chunks, 
    embeddings, # Unlike directly using chromadb, we need to define the embeddings with langchain
    collection_name="readthedocs", 
    persist_directory='llmops_s2_db',
    metadatas=all_metadata
)