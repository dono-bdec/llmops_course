from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(temperature=0,api_key=“xxx",api_version="2024-02-01",azure_endpoint="https://dono-rag-demo-resource-instance.openai.azure.com",model="GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO")
llm.invoke(‘How does ChatGPT work?’)
