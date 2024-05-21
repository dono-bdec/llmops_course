Partial Prompting with Strings

from langchain_core.prompts import PromptTemplate

# Define a prompt template with two variables
prompt = PromptTemplate.from_template("{foo}{bar}")

# Partially fill the prompt with the "foo" value
partial_prompt = prompt.partial(foo="hello")

# Later, complete the prompt with the "bar" value
print(partial_prompt.format(bar="world"))  # Output: helloworld
