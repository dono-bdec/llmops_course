# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:40:25 2024

@author: vishw
"""

from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI


llm = AzureChatOpenAI(temperature=0.1,
                      api_key="39f2089f6a2a4c52967260b0e9ffdc33",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")

class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)

misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

parser.parse(misformatted)

from langchain.output_parsers import OutputFixingParser

new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

new_parser.parse(misformatted)

