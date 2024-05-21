from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

llm = AzureChatOpenAI()

output_parser = JsonOutputParser()

format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(template="List five {subject}.\n{format_instructions}", 	input_variables=["subject"],	partial_variables={"format_instructions": format_instructions},)

chain = prompt | llm | output_parser

chain.invoke({"subject": "IPL teams"})
