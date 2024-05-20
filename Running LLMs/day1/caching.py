from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())

from datetime import datetime
start = datetime.now()
print(chat.invoke("Tell me the difference between python and R"))
end = datetime.now()
print("Elapsed time during the whole program in seconds:", end-start) 


start = datetime.now()
print(chat.invoke("Tell me the difference between python and R"))
end = datetime.now()
print("Elapsed time during the whole program in seconds:", end-start) 

