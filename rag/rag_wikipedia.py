import os
from langchain_openai import AzureChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import AzureOpenAIEmbeddings
# I DO NOT CONDONE THIS APPROACH but langchain does have an issue with some warnings at the moment
# So until they port everything correctly over to langchain_community, we will do this only for demo purposes
import warnings
warnings.filterwarnings("ignore")

CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")
azure_embeddings=""

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data
  

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks 


def create_embeddings_chroma(chunks, persist_directory=CHROMA_PATH):
    from langchain.vectorstores import Chroma

    # Instantiate an embedding model from Azure OpenAI
#VVA    embeddings = AzureOpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  
    embeddings = AzureOpenAIEmbeddings(
        model="ADA_RAG_DONO_DEMO",
        api_key="API-KEY",
        api_version="2024-02-01",
        azure_endpoint="API-ENDPOINT"
    )

    azure_embeddings = embeddings
    # Create a Chroma vector store using the provided text chunks and embedding model, 
    # configuring it to save data to the specified directory 
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory) 

    return vector_store  # Return the created vector store


def load_embeddings_chroma(persist_directory=CHROMA_PATH):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Instantiate the same embedding model used during creation
#    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536) 
    embeddings = AzureOpenAIEmbeddings(
    model="ADA_RAG_DONO_DEMO",
    api_key="API-KEY",
    api_version="2024-02-01",
    azure_endpoint="API-ENDPOINT"
    )

    # Load a Chroma vector store from the specified directory, using the provided embedding function
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings) 

    return vector_store  # Return the loaded vector store

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from datasets import Dataset 
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
    from ragas.metrics.critique import harmfulness
    from ragas import evaluate

#    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    llm = AzureChatOpenAI(temperature=0,
                      api_key="API-KEY",
                      api_version="2024-02-01",
                      azure_endpoint="API-ENDPOINT",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    PROMPT_TEMPLATE = """
Go through the context and answer given question strictly based on context. 
Context: {context}
Question: {question}
Answer:
"""
    chain = RetrievalQA.from_chain_type(llm=llm, 
                                        chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)}, 
                                        retriever=retriever)
    
    
    # answer = chain.invoke(q)
    queries = [
    "When was Lata Mangeshkar Born?",
    "Where did Lata Mangeshkar spend her childhood"    
    ]

    results = []
    for query in queries:
        result = chain({"query": query})   
        print(result)
        results.append(result['result'])

    return results
    

# Loading the pdf document into LangChain 
#data = load_document(os.path.join( os.getcwd(), 'files', 'rag_powered_by_google_search.pdf'))
data = load_from_wikipedia("Lata Mangeshkar")

# Splitting the document into chunks
chunks = chunk_data(data, chunk_size=256)

# Creating a Chroma vector store using the provided text chunks and embedding model (default is text-embedding-3-small)
vector_store = create_embeddings_chroma(chunks)

# Asking questions
q = 'When was Lata Mangeshkar born and where was her childhood spent?'
answer = ask_and_get_answer(vector_store, q)
print(answer)

