from langchain.text_splitter import MarkdownTextSplitter

# We're downloaded the readme.md file for this repo:
# https://github.com/fastfetch-cli/fastfetch


markdown_splitter = MarkdownTextSplitter(chunk_size = 500, chunk_overlap=0)

example_text=""

with open('example_data/readme_fastfetch.md', 'r') as file:
    example_text = file.read()
    
example_chunks = markdown_splitter.split_text(example_text)

for i in range(0,len(example_chunks)):
    print("-------")
    print(f"Chunk {i}: {example_chunks[i]}")