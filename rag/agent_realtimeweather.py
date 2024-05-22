import json
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv
import sys
import os

def main():

 azure_config = {
    "base_url": "<base-url>",
    "model_deployment": "<model-deployment-name>",
    "model_name": "gpt-35-turbo",
    "embedding_deployment": "<embedding-deployment-name>",
    "embedding_name": "text-embedding-ada-002",
    "api-key": "api-key",
    "api_version": "2024-02-01"
    }


#creating an Azure OpenAI client
 load_dotenv()

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

 
 initial_response = client.chat.completions.create(
    model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How is the weather in Mumbai?"}
    ],
   functions=functions
 )

 print (initial_response)
 function_name = initial_response.choices[0].message.function_call.name
 function_argument = json.loads(initial_response.choices[0].message.function_call.arguments)
 location= function_argument['location']
 if(location):
    print(location)
    getattr(sys.modules[__name__], function_name)(*function_argument)

def get_weather(location):
   #calling open weather map API for information retrieval
   #fetching latitude and longitude of the specific location respectively
    url = "http://api.openweathermap.org/geo/1.0/direct?q=" + location + "&limit=1&appid=<weather api key>"
    response=requests.get(url)
    print(response)
    get_response=response.json()
    latitude=get_response[0]['lat']
    longitude = get_response[0]['lon']

    url_final = "https://api.openweathermap.org/data/3.0/onecall?lat=" + str(latitude) + "&lon=" + str(longitude) + "&appid=<weather-api-key>"
    final_response = requests.get(url_final)
    final_response_json = final_response.json()
    #print(final_response_json)
    weather=final_response_json['current']['weather'][0]['description']
    print(weather)

if __name__ == "__main__":
    main()
 
