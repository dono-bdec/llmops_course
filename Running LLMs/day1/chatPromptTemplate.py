from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = AzureChatOpenAI()
chain = llm | StrOutputParser()

template = "Respond only in {language} language."
human_template = "Give me {object} recommendations in New Delhi."

chat_prompt = ChatPromptTemplate.from_messages([ ("system", template),  ("human", human_template),])
messages=chat_prompt.format_messages(language="Spanish", object="Restaurant")

chain.invoke(messages)
