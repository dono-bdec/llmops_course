# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
import pprint
import chromadb.utils.embedding_functions as embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    # Add the details here
)

chroma_client = chromadb.HttpClient(host="<hostname>", port="<portexposed>" )
collection = chroma_client.get_collection(name="alltextfiles_openai", embedding_function=openai_ef)

results=collection.query(query_texts=["Wilder"], n_results=3)

pprint.pprint(results)