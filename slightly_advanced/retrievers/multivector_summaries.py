from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
# I DO NOT CONDONE THIS APPROACH but langchain does have an issue with some warnings at the moment
# So until they port everything correctly over to langchain_community, we will do this only for demo purposes
import warnings
warnings.filterwarnings("ignore")


# Let's load all the blogs from all these links and then recursively split them
allLinks = ["https://paulgraham.com/superlinear.html", "https://paulgraham.com/greatwork.html", "https://paulgraham.com/read.html",
            "https://paulgraham.com/want.html", "https://paulgraham.com/users.html", "https://paulgraham.com/heresy.html"]

docs = []

for i in range(0,len(allLinks)):
    loader = WebBaseLoader(allLinks[i])
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=0, separators=[
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
    ])
original_chunks = text_splitter.split_documents(docs)
print (f"You have {len(docs)} documents that have been split into {len(original_chunks)} chunks")


# Time to summarize!
openai_llm = AzureChatOpenAI(temperature=0,
                      api_key="api_key",
                      api_version="2024-02-01",
                      azure_endpoint="api_endpoint",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")

summary_chain = load_summarize_chain(openai_llm)

# We need to use a key value pair to link between the summaries and the original document
unique_id_col = "link_id" 

# Let's store our summaries in this list
document_summaries = [] 


document_id_iter = 0
for eachChunk in original_chunks:
    #Generate an id that we can use
    unique_id = str(document_id_iter)
    document_id_iter+=1

    # Get a summary from the LLM
    chunk_summary = summary_chain.run([eachChunk])
    
    # When we get our summary, let's convert it into a document and use the unique id in the metadata
    chunk_summary_document = Document(page_content=chunk_summary, metadata={unique_id_col: unique_id})
    document_summaries.append(chunk_summary_document)

    # We also have to make sure that we use the same unique id for the metadata for the original chunk
    eachChunk.metadata[unique_id_col] = unique_id

print (f"We now get {len(document_summaries)} summaries for the original {len(original_chunks)} chunks")


embeddings = AzureOpenAIEmbeddings(
    model="ADA_RAG_DONO_DEMO",
    api_key="api_key",
    api_version="2024-02-01",
    azure_endpoint="api_endpoint"
)

# Our plain old vector store to index the summaries of the chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=embeddings)

# A storage layer for the original documents
original_documents_store = InMemoryStore()

# We now create a multi vector retriever and ensure that we use our linking key
multi_retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=original_documents_store,
    id_key=unique_id_col, 
)

# Let's add the summaries to the multi vector retriever
multi_retriever.vectorstore.add_documents(document_summaries)

# Let's search just on the summaries first
print("PRINTING SUMMARY")
_similar_docs = multi_retriever.vectorstore.similarity_search("organizations equity")
print(_similar_docs[0])


#Let's add the original documents to the retriever
multi_retriever.docstore.mset([(x.metadata[unique_id_col], x) for x in original_chunks])

retrieved_docs = multi_retriever.get_relevant_documents("organizations equity")
print("PRINTING ORIGINAL DOC")
print (retrieved_docs[0].page_content[:500])
print(len(retrieved_docs[0].page_content))
print (retrieved_docs[0].metadata)