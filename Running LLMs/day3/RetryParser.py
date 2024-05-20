# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:49:50 2024

@author: vishw
"""

from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
)
from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI

from langchain.output_parsers import RetryOutputParser


template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""

llm = AzureChatOpenAI(temperature=0.1,
                      api_key="39f2089f6a2a4c52967260b0e9ffdc33",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")

class maths(BaseModel):
    operator: str = Field(description="action to take")
    operands: list = Field(description="two random numbers")


parser = PydanticOutputParser(pydantic_object=maths)
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

prompt_value = prompt.format_prompt(query="For these two numbers:")
bad_response = '{"operator": "Add"}'

parser.parse(bad_response)

fix_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
fix_parser.parse(bad_response)


retry_parser = RetryOutputParser.from_llm(parser=parser, llm=llm)
retry_parser.parse_with_prompt(bad_response, prompt_value)

    

