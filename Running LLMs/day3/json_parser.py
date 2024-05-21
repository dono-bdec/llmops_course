# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:35:31 2024

@author: vishw
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(temperature=0.1,
                      api_key="",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")



#Without pydantic
parser = JsonOutputParser()
parser.get_format_instructions()


format_instructions = '''

Provide these attributes of a book in a json format.
{'name': name of the book,
 'author': author of the book,
 'date':year of publication,
 'pages': number of pages in the book,
 'genres': genre of the book}

'''

prompt = PromptTemplate(
    template="For this book {book}, provide the required information.\n{format_instructions}\n",
    input_variables=["book"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | model | parser

chain.invoke({"book": 'To kill a mockingbird'})



#With pydantic
# Define your desired data structure.
class book(BaseModel):
    name: str = Field(description="name of the book")
    author: str = Field(description="author of the book")
    date: int = Field(description="year of publication")
    pages: int = Field(description= "number of pages in the book")
    genres: list = Field(description = "genre of the book")
    
    

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=book)

prompt = PromptTemplate(
    template="For this book {book}, provide the required information.\n{format_instructions}\n",
    input_variables=["book"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"book": "To kill a mockingbird"})

