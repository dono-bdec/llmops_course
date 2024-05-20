import pandas as pd
import os
from datetime import datetime
from dotted_dict import DottedDict
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings


#####
##  CONFIGURATION
##
#####
azure_config = {
    "base_url": "https://dono-rag-demo-resource-instance.openai.azure.com/",
    "model_deployment": "GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO",
    "model_name": "gpt-35-turbo",
    "embedding_deployment": "ADA_RAG_DONO_DEMO",
    "embedding_name": "text-embedding-ada-002",
    "api-key": os.getenv("DONO_AZURE_OPENAI_KEY"),
    "api_version": "2024-02-01"
    }

CHROMADB_PATH = os.path.join(os.getcwd(), "chroma_db")

#####
### FUNCTIONS
#####
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain_community.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data
  
def chunk_data(documents, embedding_model):
    from langchain_experimental.text_splitter import SemanticChunker
    content=""
    for document in documents:
        content += document.page_content

    semantic_chunker = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
    return semantic_chunker.create_documents(content)

def create_models():
    models=DottedDict()
    llm = AzureChatOpenAI(temperature=0,
                      api_key=azure_config["api-key"],
                      openai_api_version=azure_config["api_version"], 
                      azure_endpoint=azure_config["base_url"],
                      model=azure_config["model_deployment"],
                      validate_base_url=False)

    models.llm=llm
    embedding_model = AzureOpenAIEmbeddings(
        api_key=azure_config["api-key"],
        openai_api_version=azure_config["api_version"],
        azure_endpoint=azure_config["base_url"],
        model = azure_config["embedding_deployment"]
    )
    models.embedding_model=embedding_model
    return  models

def ask_and_get_answers(vector_store, llm, queries, k=3):
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    ask_output = DottedDict()
    retriever = vector_store.as_retriever(search_kwargs={"k" : 1})

    PROMPT_TEMPLATE = """
Go through the context and answer given question strictly based on context. 
Context: {context}
Question: {question}
Answer:
"""
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)}, 
                                        retriever=retriever,
                                        return_source_documents=True)
    
    results = []
    contexts = []
    for query in queries:
        result = chain({"query": query})   
        results.append(result['result'])
        sources = result["source_documents"]
        contents = []
        for i in range(len(sources)):
            contents.append(sources[i].page_content)
        contexts.append(contents)
    ask_output.results = results
    ask_output.contexts = contexts

    return ask_output
    

def evaluate_results(queries, results, ground_truths, contexts, llm, embedding_model):
    from datasets import Dataset 
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
    from ragas.metrics.critique import harmfulness
    from ragas import evaluate
    
    d = {
        "question": queries,
        "answer": results,
        "contexts": contexts,
        "ground_truth": ground_truths

    }

    eval_metrics=[faithfulness, answer_relevancy, 
                                      context_precision, context_recall, context_entity_recall, 
                                      answer_similarity, answer_correctness, harmfulness]
    dataset = Dataset.from_dict(d)
    evaluation_score = evaluate(dataset,metrics=eval_metrics, llm=llm, embeddings=embedding_model)

    return evaluation_score


##############
###########   MAIN PROGRAM
##############

## Initialize models
print("Start: ", datetime.now().strftime("%d/%m/%Y:%H:%M:%S"))
print("Initializing LLM and Embedding models in Azure OpenAI...")
models = create_models()
print("Models initialized: ", datetime.now().strftime("%d/%m/%Y:%H:%M:%S"))

# Loading the data from datasource (wikipedia, pdf etc.) 
print("Obtaining documents that will be used to augment the LLM with RAG")
data = load_from_wikipedia("Lata Mangeshkar")

print("Data loaded. ", datetime.now().strftime("%d/%m/%Y:%H:%M:%S"), ". Chunking...")
# Splitting the document into chunks
chunks = chunk_data(data, models.embedding_model)

# Creating a Chroma vector store using the provided text chunks and embedding model (default is text-embedding-3-small)
print("Creating vector store that will use Azure OpenAI Embedding model () to convert the chunks into ...")
vector_store = Chroma.from_documents(
    chunks, 
    embedding=models.embedding_model, 
    collection_metadata={"hnsw:space": "cosine"}, #Change the distance function based on type of data
    persist_directory=CHROMADB_PATH,
    collection_name="wiki_lata_semantic"
    )

print("Vector store created. Documents chunked and stored in the vectordb, including embedding model to be used.")

# Asking questions
queries = [
    "When was Lata Mangeshkar Born?",
    "Where did Lata Mangeshkar move to Mumbai?",
    "What was her first Hindi song?",
    "Who did Lata Mangeshkar declare as her Godfather?"
    ]

ground_truths = [
    "November 28, 1929 in Indore",
    "She moved to Mumbai in 1945",
    "Her first hindi song was  'Mata Ek Sapoot Ki Duniya Badal De Tu'",
    "She declared Ghulam Haider as her Godfather"
    ]

print("Queries:")
print(queries)
llmchain_output = ask_and_get_answers(vector_store, models.llm, queries)
print("Results from Langchain chain (RAG+LLM):")
print(llmchain_output.results)

print("Evaluating results using RAGAS framework:")

# Evaluate answers
evaluation_score=evaluate_results(queries, llmchain_output.results, ground_truths, llmchain_output.contexts, models.llm, models.embedding_model)
score_df = evaluation_score.to_pandas()
pd.set_option('display.max_columns', None)
print(score_df)

