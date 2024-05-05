from langchain_text_splitters import CharacterTextSplitter


example_text = "Now, this is a story all about how my life got flipped and turned upside down. I'd like to take a minute so just sit right there I'll tell you how I became the prince of a town called Bel-Air"
text_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=50,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

text_chunks = text_splitter.split_text(example_text)

for i in range(0,len(text_chunks)):
    print(f"Chunk {i}: {text_chunks[i]}")