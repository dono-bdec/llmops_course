#Sequential Chains
from langchain_openai import AzureChatOpenAI
from langchain.llms import OpenAI, AzureOpenAI
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chains import SimpleSequentialChain

from langchain.llms import type_to_cls_dict

type_to_cls_dict["openai-chat"] = AzureChatOpenAI

# Define the first prompt template
prompt1 = PromptTemplate(
    input_variables=["product"],
    template="What are the key features of {product}?"
)


# Define the second prompt template
prompt2 = PromptTemplate(
    input_variables=["product"],
    template="Given the {product}, write a compelling product description for tech-savvy consumers."
)

# Initialize the language model
llm = AzureChatOpenAI(temperature=0,api_key="",api_version="2024-02-01",azure_endpoint="",model="")

# Create the first chain
chain1 = LLMChain(llm=llm, prompt=prompt1)

# Create the second chain
chain2 = LLMChain(llm=llm, prompt=prompt2)

# Combine the two chains into a sequential chain
overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

# Run the overall chain
product = "smartwatch"
output = overall_chain.run(product)
print(output)

