# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:14:12 2024

@author: vishw
"""

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)

# Define your desired data structure.
class book(BaseModel):
    name: str = Field(description="name of the book")
    author: str = Field(description="author of the book")
    date: int = Field(description="year of publication")
    pages: int = Field(description= "number of pages in the book")
    genres: list = Field(description = "genre of the book")


openai_functions = [convert_pydantic_to_openai_function(book)]

parser = JsonOutputFunctionsParser()

model = AzureChatOpenAI(temperature=0.1,
                      api_key="39f2089f6a2a4c52967260b0e9ffdc33",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful librarian. For the given book , provide the required information"), ("user", "{book}")]
)
chain = prompt | model.bind(functions=openai_functions) | parser
chain.invoke({"book": "to kill a mockingbird"})

