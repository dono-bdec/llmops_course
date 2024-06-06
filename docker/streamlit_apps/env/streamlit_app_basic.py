# Import os to handle environment variables
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.callbacks import get_openai_callback

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


st.title("Tredence RAG Demo")

with st.sidebar:
    "Docker demo"

def generate_response(input_text):

    llm = AzureChatOpenAI(temperature=0,
                      api_key=azure_config["api-key"],
                      openai_api_version=azure_config["api_version"],
                      azure_endpoint=azure_config["base_url"],
                      model=azure_config["model_deployment"],
                      validate_base_url=False)

    message = HumanMessage(
        content=input_text
    )
    
    with get_openai_callback() as cb:
        st.info(llm([message]).content) # chat model output
        st.info(cb) # callback output (like cost)

with st.form("my_form"):
    text = st.text_area("Enter text:", "What's the weather in Mumbai today?")
    submitted = st.form_submit_button("Submit")
    generate_response(text)