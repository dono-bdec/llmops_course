# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:25:34 2024

@author: vishw
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import AzureChatOpenAI

class book(BaseModel):
    
    ''' Bibliography Generator '''
    name: str = Field(description="name of the book")
    author: str = Field(description="author of the book")
    date: int = Field(description="year of publication")
    pages: int = Field(description= "number of pages in the book")
    genres: list = Field(description = "genre of the book")
    
model = AzureChatOpenAI(temperature=0.1,
                      api_key="",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO").bind_tools([book])

model.kwargs["tools"]




prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful librarian. For the given book , provide the required information"), ("user", "{book}")]
)
from langchain.output_parsers.openai_tools import JsonOutputToolsParser

parser = JsonOutputToolsParser()
chain = prompt | model | parser
x=chain.invoke({"book": "1984"})

x[0]['args']['name']

x[0]['args']['author']
