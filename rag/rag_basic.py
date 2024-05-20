from datasets import Dataset 
import pandas as pd
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
from langchain_openai import AzureChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import AzureOpenAIEmbeddings

azure_config = {
    "base_url": "https://dono-rag-demo-resource-instance.openai.azure.com/",
    "model_deployment": "GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO",
    "model_name": "gpt-35-turbo",
    "embedding_deployment": "ADA_RAG_DONO_DEMO",
    "embedding_name": "text-embedding-ada-002",
    "api-key": os.getenv("DONO_AZURE_OPENAI_KEY"),
    "api_version": "2024-02-01"
    }

llm = AzureChatOpenAI(temperature=0,
                      api_key=azure_config["api-key"],
                      openai_api_version=azure_config["api_version"],
                      azure_endpoint=azure_config["base_url"],
                      model=azure_config["model_deployment"],
                      validate_base_url=False)

embeddings = AzureOpenAIEmbeddings(
        api_key=azure_config["api-key"],
        openai_api_version=azure_config["api_version"],
        azure_endpoint=azure_config["base_url"],
        model = azure_config["embedding_deployment"]
    )

sample_dataset = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}

dataset = Dataset.from_dict(sample_dataset)
eval_metrics=[faithfulness, answer_relevancy, 
                                      context_precision, context_recall, context_entity_recall, 
                                      answer_similarity, answer_correctness]
   
score = evaluate(dataset,metrics=eval_metrics, llm=llm, embeddings=embeddings)
score_df=score.to_pandas()
pd.set_option('display.max_columns', None)
print(score_df)
