# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:06:16 2024

@author: vishw
"""

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "This action-packed thriller kept me on the edge of my seat from start to finish! The stunts were incredible, and the storyline had me hooked. Definitely a must-watch for any action movie fan!", "output": "Action"},
    {"input": "I havent laughed this hard in ages!", "output": "Comedy"},
    {"input": "A truly moving story that tugs at the heartstrings. The characters were well-developed, and the emotional depth of the film left a lasting impact. ", "output": "Drama"},
    {"input": "I watched this movie with all the lights on! The suspense was unbearable, and the scares were genuinely terrifying. If you're a fan of horror, this film will definitely give you chills down your spine.", "output": "Horror"},
    {"input": " fell in love with this movie! The chemistry between the leads was palpable, and the romantic storyline was beautifully portrayed.", "output": "Romance"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

embeddings = AzureOpenAIEmbeddings(

    model="ADA_RAG_DONO_DEMO",

    api_key="",

    api_version="2024-02-01",

    azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com"

)


example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # The list of examples available to select from.
    examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    embeddings,
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # The number of examples to produce.
    k=2,
)

mmr_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Guess the genre of the movie by the review",
    suffix="Input: {review}\nOutput:",
    input_variables=["review"],
)

print(mmr_prompt.format(review="This mind-bending sci-fi adventure took me on an exhilarating journey through space and time! "))

