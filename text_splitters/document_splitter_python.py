from langchain.text_splitter import PythonCodeTextSplitter


python_splitter = PythonCodeTextSplitter(chunk_size=500, chunk_overlap=0)

example_text=""

with open('example_data/sample_python.py', 'r') as file:
    example_text = file.read()
    
example_chunks = python_splitter.split_text(example_text)

for i in range(0,len(example_chunks)):
    print("-------")
    print(f"Chunk {i}: {example_chunks[i]}")