# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 07:35:56 2024

@author: vishw
"""

from langchain_core.prompts import PromptTemplate

# Define a prompt template with two variables
prompt = PromptTemplate.from_template("{foo}{bar}")

# Partially fill the prompt with the "foo" value
partial_prompt = prompt.partial(foo="hello")

# Later, complete the prompt with the "bar" value
print(partial_prompt.format(bar="world"))  # Output: helloworld

from langchain import hub
prompt = hub.pull("pollywantsapualie/superb_system_instruction_prompt")
