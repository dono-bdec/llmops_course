from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

example_text=""


with open('example_data/sample_js.js', 'r') as file:
    example_text = file.read()

# You can see the full list of languages supported in the Language import
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=200, chunk_overlap=0
)

example_chunks = js_splitter.split_text(example_text)

for i in range(0,len(example_chunks)):
    print("-------")
    print(f"Chunk {i}: {example_chunks[i]}")



