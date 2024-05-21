# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:08:06 2024

@author: vishw
"""

from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(temperature=0.1,
                      api_key="",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")

output_parser = DatetimeOutputParser()
#print(output_parser.get_format_instructions())
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="Generate a paragraph on the biography of {person} and provide their date of birth.\n{format_instructions}",
    input_variables=["person"],
    partial_variables={"format_instructions": format_instructions},
)


chain = prompt | llm | output_parser

output = chain.invoke({"person": "Alan turing"})

print(output)


