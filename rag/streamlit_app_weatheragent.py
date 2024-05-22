import sys
import json
import os
import requests
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI
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
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_config["base_url"]
    #openai_api_key = st.text_input("OpenAI API Key", type="password") 
    os.environ["AZURE_OPENAI_API_KEY"] = azure_config["api-key"]
    "[Get an Azure OpenAI API key from 'Keys and Endpoint' in Azure Portal](https://portal.azure.com/#blade/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/OpenAI)"

def generate_response(input_text):

    client = AzureOpenAI(
        azure_endpoint = azure_config["base_url"], 
        api_key=azure_config["api-key"],
        api_version=azure_config["api_version"]
        )
    functions=[
        {
            "name":"get_weather",
            "description":"Retrieve real-time weather information/data about a particular location/place",
            "parameters":{
                "type":"object",
                "properties":{
                    "location":{
                        "type":"string",
                        "description":"the exact location whose real-time weather is to be determined",
                    },
                    
                },
                "required":["location"]
            },
        }
    ] 

    with st.spinner('Processing...'):
        initial_response = client.chat.completions.create(
            model=azure_config["model_deployment"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            functions=functions
        )
        if (initial_response.choices[0].finish_reason == 'function_call'):
            function_name = initial_response.choices[0].message.function_call.name
            function_argument = json.loads(initial_response.choices[0].message.function_call.arguments)
            location= function_argument['location']
            if(location):
                resp=getattr(sys.modules[__name__], function_name)(location)
        else:
            resp=initial_response.choices[0].message.content

    #st.success('Done')
    #print (initial_response)

    with get_openai_callback() as cb:
        st.info(resp)
#        for line in weather_result.splitlines():
#            st.info(line) # chat model output
        st.info(cb) # callback output (like cost)

def get_weather(location):
    #print("getting weather for: " + location)
    #calling open weather map API for information retrieval
    #fetching latitude and longitude of the specific location respectively
    url = "http://api.openweathermap.org/geo/1.0/direct?q=" + location + "&limit=1&appid="+os.getenv("OPENWEATHERMAP_API_KEY")
    response=requests.get(url)
    get_response=response.json()
    latitude=get_response[0]['lat']
    longitude = get_response[0]['lon']

    url_final = "https://api.openweathermap.org/data/3.0/onecall?lat=" + str(latitude) + "&lon=" + \
        str(longitude) + "&appid=" + os.getenv("OPENWEATHERMAP_API_KEY")
    final_response = requests.get(url_final)
    final_response_json = final_response.json()
    weather="Weather for " + location + ":  \n" + final_response_json['current']['weather'][0]['description'] + "  \nHumidity: " + str(final_response_json['current']['humidity']) + "  \nUV Index: " + str(final_response_json['current']['uvi'])
    return weather

with st.form("my_form"):
    text = st.text_area("Enter text:", "What's the weather in Mumbai today?")
    submitted = st.form_submit_button("Submit")
    generate_response(text)