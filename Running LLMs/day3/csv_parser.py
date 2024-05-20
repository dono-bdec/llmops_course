# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:53:54 2024

@author: vishw
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

llm = AzureChatOpenAI(temperature=0.1,
                      api_key="39f2089f6a2a4c52967260b0e9ffdc33",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

#Simple text generation
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(template="List five {subject}.\n{format_instructions}",  input_variables=["subject"], partial_variables={"format_instructions": format_instructions},)

chain = prompt | llm | output_parser
chain.invoke({"subject": "IPL teams"})


#Analysing larger responses 
format_instructions = ''' Analyse the text on biographies of famous persons given along and only provide these details in a comma separated schema:
    "name", name of person
    "date of birth", date of birth of the said person
    "Education", name of institution where the person studied,
    "date of death", date when the person died
    Do not provide anything else than this.
    '''

prompt = PromptTemplate(
    template="Generate a paragraph on the biography of {person}.\n{format_instructions}",
    input_variables=["person"],
    partial_variables={"format_instructions": format_instructions},
)
prompt = PromptTemplate(
    template="Generate a paragraph on the biography of {person}.",
    input_variables=["person"],
)

chain = prompt | llm | output_parser
chain.invoke({"person": "Alan Turing"})

