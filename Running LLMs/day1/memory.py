from langchain import OpenAI, ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize the memory object to store conversation history
memory = ConversationalBufferMemory()

# Initialize the language model
llm = OpenAI(temperature=0)

# Create the conversation chain with the memory object
conversation = ConversationChain( llm=llm, memory=memory, verbose=True)

# Start the conversation
print('Human: Hi there!')
response = conversation.run('Hi there!')
print(f'AI: {response}')

# The memory will store the previous interaction
print('Human: What is the capital of France?')
response = conversation.run('What is the capital of France?')
print(f'AI: {response}')

# The model can now reference the conversation history
print('Human: And what is the currency used there?')
response = conversation.run('And what is the currency used there?')
print(f'AI: {response}')
