# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
import chromadb


chromadbClient = chromadb.HttpClient(host="127.0.0.1", port="1201")

all_text_files = glob.glob('sample_text_files/*.txt')

example_data = []

for i in range(0, len(all_text_files)):
    with open(all_text_files[i], 'r') as file:
        example_data.append(file.read())

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
)

all_documents=[]


# You can also provide a list of data here. 
# I've taken the longer route with all this code so that it's easier to understand how we are ingesting here
for i in range(0,len(example_data)):
    curr_recursive_split_chunks = recursive_text_splitter.split_text(example_data[i])
    all_documents.append(curr_recursive_split_chunks)

all_docs = []
all_metadata=[]
for i in range(0,len(all_documents)):
    for each_doc in all_documents[i]:
        all_docs.append(each_doc)
        all_metadata.append({"reference": all_text_files[i]})


embeddings = AzureOpenAIEmbeddings(
    # Add the details here
)

store = Chroma.from_texts(
    all_docs, 
    embeddings, # Unlike directly using chromadb, we need to define the embeddings with langchain
    collection_name="alltextfiles_openai", 
    metadatas=all_metadata,
    client=chromadbClient,
)
