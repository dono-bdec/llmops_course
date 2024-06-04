import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI

import warnings
warnings.filterwarnings("ignore")

CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")
azure_embeddings=""
load_dotenv()

azure_config = {
    "base_url": os.getenv("DONO_AZURE_OPENAI_BASE_URL"),
    "model_deployment": os.getenv("DONO_AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
    "model_name": os.getenv("DONO_AZURE_OPENAI_MODEL_NAME"),
    "embedding_deployment": os.getenv("DONO_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    "embedding_name": os.getenv("DONO_AZURE_OPENAI_EMBEDDING_NAME"),
    "api-key": os.getenv("DONO_AZURE_OPENAI_API_KEY"),
    "api_version": os.getenv("DONO_AZURE_OPENAI_API_VERSION")
    }


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
    embeddings = AzureOpenAIEmbeddings(
        api_key=os.getenv("DONO_AZURE_OPENAI_API_KEY"),
        openai_api_version=azure_config["api_version"],
        azure_endpoint=azure_config["base_url"],
        azure_deployment=azure_config["embedding_deployment"],
        model = azure_config["embedding_name"]
    )

    # Create a Chroma vector store using the provided text chunks and embedding model, 
    # configuring it to save data to the specified directory 
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory) 

    return vector_store  # Return the created vector store


def load_embeddings_chroma(persist_directory=CHROMA_PATH):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Instantiate the same embedding model used during creation
    embeddings = AzureOpenAIEmbeddings(
        model=azure_config["embedding_deployment"],
        api_version=azure_config["api_version"],
        azure_endpoint=azure_config["base_url"]
    )

    # Load a Chroma vector store from the specified directory, using the provided embedding function
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings) 

    return vector_store  # Return the loaded vector store

def ask_and_get_answer(vector_store, q, k=3):
    client = AzureOpenAI(
        azure_endpoint=azure_config["base_url"],
        api_key=azure_config["api-key"],
        api_version="2024-02-01"
    )
    model_name=azure_config["model_deployment"]
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    user_input = q
 
    # Design prompts
    qna_system_message = """
    You are an assistant to a financial services firm who answers user queries on annual reports.
    User input will have the context required by you to answer user questions.
    This context will begin with the token: ###Context.
    The context contains references to specific portions of a document relevant to the user query.

    User questions will begin with the token: ###Question.

    Please answer user questions only using the context provided in the input.
    Do not mention anything about the context in your final answer. Your response should only contain the answer to the question.

    If the answer is not found in the context, respond "I don't know".
    """
    qna_user_message_template = """
    ###Context
    Here are some documents that are relevant to the question mentioned below.
    {context}

    ###Question
    {question}
    """
    relevant_document_chunks = retriever.get_relevant_documents(user_input)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)

    prompt = [
        {'role':'system', 'content': qna_system_message},
        {'role': 'user', 'content': qna_user_message_template.format(
            context=context_for_query,
            question=user_input
            )
        }
    ]

    relevant_document_chunks = retriever.get_relevant_documents(user_input)

    print("length of relevant document chunks: " + str(len(relevant_document_chunks)))
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            temperature=0
        )
        prediction = response.choices[0].message.content.strip()
    except Exception as e:
        prediction = f'Sorry, I encountered the following error: \n {e}'
    
    print(prediction)
    return prediction
    
def evaluate_answer(retriever, q, answer):
    # Evaluation
    # Let us now use the LLM-as-a-judge method to check the quality of the RAG system
    # on two parameters - retrieval and generation. 
    # We illustrate this evaluation based on the answeres generated to the question 
    # from the previous section.

    rater_model = 'gpt-35-turbo' # 'gpt-4'

    groundedness_rater_system_message = """
    You are tasked with rating AI generated answers to questions posed by users.
    You will be presented a question, context used by the AI system to generate the answer and an AI generated answer to the question.
    In the input, the question will begin with ###Question, the context will begin with ###Context while the AI generated answer will begin with ###Answer.

    Groundedness evaluation criteria:
    The task is to judge the extent to which the metric is followed by the answer.
    1 - The metric is not followed at all
    2 - The metric is followed only to a limited extent
    3 - The metric is followed to a good extent
    4 - The metric is followed mostly
    5 - The metric is followed completely

    Metric:
    Groundedness measures how well the answer is based on the information presented in the context.
    The answer should be derived only from the information presented in the context

    Instructions:
    1. First write down the steps that are needed to evaluate the answer as per the metric.
    2. Give a step-by-step explanation if the answer adheres to the metric considering the question and context as the input.
    3. Next, evaluate the extent to which the metric is followed.
    4. Use the previous information to rate the answer using the groundedness evaluaton criteria and assign a groundedness score.
    """

    relevance_rater_system_message = """
    You are tasked with rating AI generated answers to questions posed by users.
    You will be presented a question, context used by the AI system to generate the answer and an AI generated answer to the question.
    In the input, the question will begin with ###Question, the context will begin with ###Context while the AI generated answer will begin with ###Answer.

    Relevance evaluation criteria:
    The task is to judge the extent to which the metric is followed by the answer.
    1 - The metric is not followed at all
    2 - The metric is followed only to a limited extent
    3 - The metric is followed to a good extent
    4 - The metric is followed mostly
    5 - The metric is followed completely

    Metric:
    Relevance measures how well the answer addresses the main aspects of the question, based on the context.
    Consider whether all and only the important aspects are contained in the answer when evaluating relevance.

    Instructions:
    1. First write down the steps that are needed to evaluate the context as per the metric.
    2. Give a step-by-step explanation if the context adheres to the metric considering the question as the input.
    3. Next, evaluate the extent to which the metric is followed.
    4. Use the previous information to rate the context using the relevance evaluaton criteria and assign a Relevance score.
    """

    user_message_template = """
    ###Question
    {question}

    ###Context
    {context}

    ###Answer
    {answer}
    """
    qna_system_message = """
    You are an assistant to a financial services firm who answers user queries on annual reports.
    User input will have the context required by you to answer user questions.
    This context will begin with the token: ###Context.
    The context contains references to specific portions of a document relevant to the user query.

    User questions will begin with the token: ###Question.

    Please answer user questions only using the context provided in the input.
    Do not mention anything about the context in your final answer. Your response should only contain the answer to the question.

    If the answer is not found in the context, respond "I don't know".
    """
    qna_user_message_template = """
    ###Context
    Here are some documents that are relevant to the question mentioned below.
    {context}

    ###Question
    {question}
    """
    user_input = q
    relevant_document_chunks = retriever.get_relevant_documents(user_input)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)

    prompt = [
        {'role':'system', 'content': qna_system_message},
        {'role': 'user', 'content': qna_user_message_template.format(
            context=context_for_query,
            question=user_input
            )
        }
    ]

    # Ideally should be another model instance
    model_name=azure_config["model_deployment"]
    client = AzureOpenAI(
        azure_endpoint=azure_config["base_url"],
        api_key=azure_config["api-key"],
        api_version="2024-02-01"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    print(answer)
    groundedness_prompt = [
        {'role':'system', 'content': groundedness_rater_system_message},
        {'role': 'user', 'content': user_message_template.format(
            question=user_input,
            context=context_for_query,
            answer=answer
            )
        }
    ]

    #Ideally should be a different model
    rater_model=azure_config["model_deployment"]
    response = client.chat.completions.create(
        model=rater_model,
        messages=groundedness_prompt,
        temperature=0
    )

    print(response.choices[0].message.content)

    relevance_prompt = [
        {'role':'system', 'content': relevance_rater_system_message},
        {'role': 'user', 'content': user_message_template.format(
            question=user_input,
            context=context_for_query,
            answer=answer
            )
        }
    ]

    response = client.chat.completions.create(
        model=rater_model,
        messages=relevance_prompt,
        temperature=0
    )

    print(response.choices[0].message.content)


### MAIN FUNCTION
# Loading the pdf document into LangChain 
#data = load_document(os.path.join( os.getcwd(), 'files', 'rag_powered_by_google_search.pdf'))
data = load_from_wikipedia("Lata Mangeshkar")

# Splitting the document into chunks
chunks = chunk_data(data, chunk_size=256)

# Creating a Chroma vector store using the provided text chunks and embedding model (default is text-embedding-3-small)
vector_store = create_embeddings_chroma(chunks)
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)
# Asking questions
q = 'When was Lata Mangeshkar born?'
answer = ask_and_get_answer(vector_store, q)

# Evaluate response
evaluate_answer(retriever, q, answer)

