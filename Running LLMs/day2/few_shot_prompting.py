# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:06:41 2024

@author: vishw
"""

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [ { "question": "Who composed the soundtrack for the movie 'Inception'?",
 "answer": """ Are follow up questions needed here: Yes.
Follow up: What is the name of the composer?
Intermediate answer: The composer of the soundtrack for 'Inception' is Hans Zimmer.
Follow up: Is Hans Zimmer known for composing music for other movies as well?
Intermediate answer: Yes, Hans Zimmer is a renowned composer who has composed music for various other movies including 'The Dark Knight', 'Gladiator', and 'Interstellar'.
So the final answer is: Hans Zimmer"""},
{"question": "Who is the author of 'To Kill a Mockingbird'?",
  "answer":"""Are follow up questions needed here: Yes.
Follow up: What is the name of the author?
Intermediate answer: The author of 'To Kill a Mockingbird' is Harper Lee.
Follow up: Did Harper Lee write any other notable works?
Intermediate answer: Yes, Harper Lee wrote another novel titled 'Go Set a Watchman', which was published after 'To Kill a Mockingbird’.
So the final answer is: Harper Lee"""},

{"question": "What is the capital city of Australia?",
"answer": """ Are follow up questions needed here: No.
Final answer: Canberra"""}]

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")
print(example_prompt.format(**examples[0]))


prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

print(prompt.format(input="When did World War 1 end?"))

from langchain_openai import AzureChatOpenAI
chat = AzureChatOpenAI(temperature=0,api_key="",api_version="2024-02-01",azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")

chat.invoke(prompt.format(input="When did World War 1 end?"))
