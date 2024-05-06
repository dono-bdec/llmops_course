# Uncomment this code if you're running this on the Lab VM
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
import pprint
import chromadb.utils.embedding_functions as embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="ADA_RAG_DONO_DEMO",
                api_key=pinginstructorforkey,
                api_version="2024-02-01",
                api_base=pinginstructorforurl,
                api_type="azure"
            )

chroma_client = chromadb.PersistentClient(path="llmops_s3_db")
collection = chroma_client.get_collection(name="alltextfiles_openai", embedding_function=openai_ef)

results=collection.query(query_texts=["Emirates"], n_results=3)

pprint.pprint(results)