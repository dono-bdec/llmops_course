# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:45:21 2024

@author: vishw
"""

import pprint
from typing import Any, Dict

import pandas as pd
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(temperature=0.1,
                      api_key="",
                      api_version="2024-02-01",
                      azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",
                      model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")

# Solely for documentation purposes.
def format_parser_output(parser_output: Dict[str, Any]) -> None:
    for key in parser_output.keys():
        parser_output[key] = parser_output[key].to_dict()
    return pprint.PrettyPrinter(width=4, compact=True).pprint(parser_output)

# Define your desired Pandas DataFrame.
df = pd.DataFrame(
    {
        "num_legs": [2, 4, 8, 0],
        "num_wings": [2, 0, 0, 0],
        "num_specimen_seen": [10, 2, 1, 8],
    }
)

# Set up a parser + inject instructions into the prompt template.
parser = PandasDataFrameOutputParser(dataframe=df)

# Here's an example of a column operation being performed.
df_query = "Retrieve the num_wings column."

# Set up the prompt.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

'''
The output should be formatted as a string as the operation, 
followed by a colon, followed by the column or row to be queried on, 
followed by optional array parameters.\n1. 
The column names are limited to the possible columns below.\n2. 
Arrays must either be a comma-separated list of numbers formatted as [1,3,5], 
or it must be in range of numbers formatted as [0..4].\n3. 
Remember that arrays are optional and not necessarily required.\n4. 
If the column is not in the possible columns or the operation is not a valid Pandas 
DataFrame operation, return why it is invalid as a sentence starting with either 
"Invalid column" or "Invalid operation".\n\nAs an example, for the formats:\n1. 
String "column:num_legs" is a well-formatted instance which gets the column num_legs, 
where num_legs is a possible column.\n2. String "row:1" is a well-formatted instance 
which gets row 1.\n3. String "column:num_legs[1,2]" is a well-formatted instance 
which gets the column num_legs for rows 1 and 2, where num_legs is a possible column.
\n4. String "row:1[num_legs]" is a well-formatted instance which gets row 1, 
but for just column num_legs, where num_legs is a possible column.\n5. 
String "mean:num_legs[1..3]" is a well-formatted instance which takes 
the mean of num_legs from rows 1 to 3, where num_legs is a possible 
column and mean is a valid Pandas DataFrame operation.\n6.
 String "do_something:num_legs" is a badly-formatted instance, 
 where do_something is not a valid Pandas DataFrame operation.\n7. 
 String "mean:invalid_col" is a badly-formatted instance, 
 where invalid_col is not a possible column.
 \n\nHere are the possible columns:\n```\nnum_legs, num_wings, num_specimen_seen\n```\n'
'''

chain = prompt | model | parser
parser_output = chain.invoke({"query": df_query})

format_parser_output(parser_output)
{'mean': 4.0}
# Here's an example of a row operation being performed.
df_query = "Retrieve the first row."

# Set up the prompt.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
parser_output = chain.invoke({"query": df_query})

format_parser_output(parser_output)



# Here's an example of a random Pandas DataFrame operation limiting the number of rows
df_query = "Retrieve the average of the num_legs column from rows 1 to 3."

# Set up the prompt.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
parser_output = chain.invoke({"query": df_query})

print(parser_output)

# Here's an example of a poorly formatted query
df_query = "Retrieve the mean of the num_fingers column."

# Set up the prompt.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
parser_output = chain.invoke({"query": df_query})
