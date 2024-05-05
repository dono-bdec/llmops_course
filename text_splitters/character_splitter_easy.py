example_text = "Now, this is a story all about how my life got flipped and turned upside down. I'd like to take a minute so just sit right there I'll tell you how I became the prince of a town called Bel-Air"
chunk_size = 50

text_chunks = []
chunk_overlap = 0

for i in range(0, len(example_text), (chunk_size-chunk_overlap)):
    chunk = example_text[i:i+chunk_size]
    text_chunks.append(chunk)

for i in range(0,len(text_chunks)):
    print(f"Chunk {i}: {text_chunks[i]}")