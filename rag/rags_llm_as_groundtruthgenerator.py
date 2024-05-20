from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.text_splitter import CharacterTextSplitter
from langfuse.callback import CallbackHandler
from azure.search.documents.indexes.models import (
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)
from tqdm import tqdm
from langfuse import Langfuse
from datasets import Dataset  
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision,context_recall
from ragas.metrics.critique import SUPPORTED_ASPECTS, harmfulness
import pandas as pd
import nest_asyncio
nest_asyncio.apply()
import warnings
warnings.filterwarnings("ignore") 

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


def split_doc(filename_):
    print(f'Reading - {filename_}')
    loader = TextLoader(filename_, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs

def add_metadata(data,time):
    for chunk in data:
        chunk.metadata['last_update'] = time
    return data

msft_q1 = split_doc('MSFT_q1_2024.txt')
msft_q2 = split_doc('MSFT_q2_2024.txt')

# Adding same data with different last_update 
from datetime import datetime, timedelta

q2_time = (datetime.utcnow() - timedelta(days=90)).strftime(
    "%Y-%m-%dT%H:%M:%S-00:00"
)
q1_time = (datetime.utcnow() - timedelta(days=180)).strftime(
    "%Y-%m-%dT%H:%M:%S-00:00"
)


msft_q1 = add_metadata(msft_q1,q1_time)
msft_q2 = add_metadata(msft_q2,q2_time)

documents = msft_q1 + msft_q2

print(len(documents))

embeddings = AzureOpenAIEmbeddings(
        api_key=azure_config["api-key"],
        openai_api_version=azure_config["api_version"],
        azure_endpoint=azure_config["base_url"],
        model = azure_config["embedding_deployment"]
    )


embedding_function=embeddings.embed_query

fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=len(embedding_function("Text")),
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field for filtering on document source
    SimpleField(
        name="source",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
    # Additional data field for last doc update
    SimpleField(
        name="last_update",
        type=SearchFieldDataType.DateTimeOffset,
        searchable=True,
        filterable=True,
    ),
]
# Adding a custom scoring profile with a freshness function
sc_name = "scoring_profile"
sc = ScoringProfile(
    name=sc_name,
    text_weights=TextWeights(weights={"content": 5}),
    function_aggregation="sum",
    functions=[
        FreshnessScoringFunction(
            field_name="last_update",
            boost=100,
            parameters=FreshnessScoringParameters(boosting_duration="P2D"),
            interpolation="linear",
        )
    ],
)

index_name = "earning_call-scoring-profile"

# vector_store: AzureSearch = AzureSearch(
#     azure_search_endpoint=azure_config["base_url"],
#     azure_search_key=azure_config["api-key"],
#     index_name=index_name,
#     embedding_function=embeddings.embed_query,
#     fields=fields,
#     scoring_profiles=[sc],
#     default_scoring_profile=sc_name,
# )

# azureai_retriever = vector_store.as_retriever()

# azureai_retriever.invoke("How is Windows OEM revenue growth?")

from ragas.testset.generator import TestsetGenerator

llm = AzureChatOpenAI(temperature=0,
                      api_key=azure_config["api-key"],
                      openai_api_version=azure_config["api_version"], 
                      azure_endpoint=azure_config["base_url"],
                      model=azure_config["model_deployment"],
                      validate_base_url=False)

generator_llm = llm
critic_llm = llm

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

#generating testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 1},is_async=False )
testset.to_pandas().to_excel('Ground_Truth_Dataset.xlsx',index=False)#['question'].tolist()


