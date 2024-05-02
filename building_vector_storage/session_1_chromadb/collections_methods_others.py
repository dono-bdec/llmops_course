import chromadb


chroma_client = chromadb.PersistentClient(path="llmops_s1")


collection = chroma_client.get_collection("my_collection")

print(collection.count())


pprint.pprint(collection.peek())