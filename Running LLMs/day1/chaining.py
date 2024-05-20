from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

llm = AzureChatOpenAI()
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(template="List five {subject}.\n{format_instructions}", 	input_variables=["subject"],	partial_variables={"format_instructions": format_instructions},)

#CSV output parser
output_parser = CommaSeparatedListOutputParser()

#Chaining
chain = prompt | llm | output_parser
chain.invoke({"subject": "IPL teams"})


#Change the output parser to jsonoutputparser without changing rest of the code
from langchain_core.output_parsers import JsonOutputParser
output_parser = JsonOutputParser()
chain = prompt | llm | output_parser

chain.invoke({"subject": "IPL teams"})
