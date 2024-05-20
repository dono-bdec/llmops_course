# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:48:01 2024

@author: vishw
"""

from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

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
example_selector = LengthBasedExampleSelector(
    # The examples it has available to choose from.
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    # The maximum length that the formatted examples should be.
    # Length is measured by the get_text_length function below.
    max_length=20,
    # The function used to get the length of a string, which is used
    # to determine which examples to include. It is commented out because
    # it is provided as a default value if none is specified.
    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
)
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Guess the genre of the movie by the review",
    suffix="Input: {review}\nOutput:",
    input_variables=["review"],
)

print(dynamic_prompt.format(review="This mind-bending sci-fi adventure took me on an exhilarating journey through space and time! "))

