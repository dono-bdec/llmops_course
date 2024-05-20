import random
from langchain.prompts.prompt import PromptTemplate



def animal():
	return random.choice(["cow", "dog","chicken", 'seal'])

prompt = PromptTemplate(
    template="Why did {animal} cross the {location}?",
    input_variables=["animal", "location"],
)
partial_prompt = prompt.partial(animal = animal)
print(partial_prompt.format(location="street"))
