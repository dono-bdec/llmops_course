# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:18:23 2024

@author: vishw
"""

from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(temperature=0.1,
                      api_key="",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")


# Define your desired data structure.
class book(BaseModel):
    name: str = Field(description="name of the book")
    author: str = Field(description="author of the book")
    date: int = Field(description="year of publication")
    pages: int = Field(description= "number of pages in the book")
    genres: list = Field(description = "genre of the book")
   
    # You can add custom validation logic easily with Pydantic.
    @validator("date")
    def genres_fiction(cls, field):
        if field > 1990:
            raise ValueError("Recent book. Not a classic")
        return field



# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=book)
# Set up the prompt.
prompt = PromptTemplate(
    template="You are a librarian. Given {book}, provide these required details\n{format_instructions}\n",
    input_variables=["book"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


chain = prompt | model | parser

#chain.invoke({"book": "To kill a mocking bird"})


chain.invoke({"book": "Harry Potter and the Prisoner of Azkaban"})

chain.invoke({"book": "To kill a mockingbird"})
