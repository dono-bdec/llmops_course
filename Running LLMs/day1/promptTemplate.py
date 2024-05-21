from langchain.prompts import PromptTemplate

# Define a prompt template
template = "What is the meaning of the word {word}?"

# Create a prompt using the template
prompt = PromptTemplate(template=template)

# Generate a prompt instance with a value for the placeholder
prompt_instance = prompt.format_prompt(word="chatbot")

# Print the generated prompt
print(prompt_instance)
