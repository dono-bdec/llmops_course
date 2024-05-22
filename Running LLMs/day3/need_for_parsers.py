# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:47:53 2024

@author: vishw
"""


from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = AzureChatOpenAI(temperature=0.1,
                      api_key="",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")

parser = StrOutputParser() #Converts AIMessage content to string

prompt = ChatPromptTemplate.from_template("Give me {number} random numbers ")

chain = prompt | llm | parser
numbers = chain.invoke({'number':10})
print([int(x) for x in numbers.split(',')])
numbers= [int(x) for x in numbers.split(',')]
if (42 in numbers):
    print('wow')
else:
    print('not wow')

numbers = chain.invoke({'number':2})
print([int(x) for x in numbers.split(',')])
numbers= [int(x) for x in numbers.split(',')]
if (42 in numbers):
    print('wow')
else:
    print('not wow')


#chain.batch([{'number':10}, {'number':2},  {'number':5}])

