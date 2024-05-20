# -*- coding: utf-8 -*-
"""
Created on Thu May  2 00:09:57 2024

@author: vishw
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage

llm = AzureChatOpenAI(temperature=0.1,
                      api_key="39f2089f6a2a4c52967260b0e9ffdc33",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")



def parse(ai_msg: AIMessage) -> str:
    """Parse the AI message."""
    x= ai_msg.content.split(' ')
    return ':-) '.join(x)


chain = llm | parse
print(chain.invoke("Hi"))
