# -*- coding: utf-8 -*-
"""
Sentiment Analysis with LLM

Learning Objectives

Use few-shot prompting to build LLM applications for classification tasks.
"""

import json
import numpy as np
import os
from openai import AzureOpenAI
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()

azure_config = {
    "base_url": os.getenv("DONO_AZURE_OPENAI_BASE_URL"),
    "model_deployment": os.getenv("DONO_AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
    "model_name": os.getenv("DONO_AZURE_OPENAI_MODEL_NAME"),
    "embedding_deployment": os.getenv("DONO_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    "embedding_name": os.getenv("DONO_AZURE_OPENAI_EMBEDDING_NAME"),
    "api-key": os.getenv("DONO_AZURE_OPENAI_API_KEY"),
    "api_version": os.getenv("DONO_AZURE_OPENAI_API_VERSION")
    }


client = AzureOpenAI(
    azure_endpoint=azure_config["base_url"],
    api_key=azure_config["api-key"],
    api_version="2024-02-01"
)
model_name=azure_config["model_deployment"]

"""
Business Use Case

Most organizations mandate product managers to constantly monitor product reviews on ecommerce sites such as Amazon to
 digest #user feedback and gain insights into product adoption. Given the volume of reviews that a product team has to 
 deal with, sifting #through reviews and flagging those that warrant a response is a human-intensive task that is ripe 
 for automation. In this #context, assigning sentiment to product reviews (sentiment analysis) and identifying specific 
 aspects of the product and the #associated sentiment (aspect-based sentiment analysis) become critical.

Beyond analysing sentiment attached to a product review, some additional interesting questions that could be answered using LLMs #are:
- Features that are liked and disliked the most
- Fraction of reviews that express extreme dissatisfaction
- Whether the product is deemed to be value-for-money?

Sentiment Analysis

Prompt Design

We will design a few-shot prompt for sentiment analysis. We first assemble a set of examples (input-output exemplars) 
as a part of the few-shot prompt. Then we create the prompt in the Azure Open API format.

To evaluate the performance of the *prompt* we will use a set of *gold examples*, that is, a set of examples that is 
not presented to the model as a part of the prompt.

**Examples and Gold Examples**

#A set of examples and gold examples for sentiment classification of Amazon product reviews is hosted in a HuggingFace dataset. 
Let us load this data and take a look at the samples in this data.

amazon_reviews = load_dataset("pgurazada1/amazon_polarity")

#As is evident from the above output, the data set has 32 samples as examples and 32 samples as gold examples.

"""
amazon_reviews = load_dataset("pgurazada1/amazon_polarity")
amazon_reviews_examples_df = amazon_reviews['examples'].to_pandas()
amazon_reviews_gold_examples_df = amazon_reviews['gold_examples'].to_pandas()

amazon_reviews_examples_df.shape, amazon_reviews_gold_examples_df.shape

"""As the above outputs indicate, there are 32 examples and 32 gold examples. We will sample from the examples to create the few shot prompt and evaluate the prompt on all 32 gold examples."""

amazon_reviews_examples_df.sample(4)

"""**Assembling the prompt**"""

system_message = """
Classify product reviews in the input as positive or negative in sentiment.
Do not explain your answer. Your answer should only contain the label: 1 (positive) or 0 (negative).
"""

few_shot_prompt = [{'role':'system', 'content': system_message}]

""" We need to iterate over the rows of the examples DataFrame to append these examples as `user` and `assistant` 
    messages to the few-shot prompt. We achieve this using the `iterrows` method.
"""

for index, row in amazon_reviews_examples_df.iterrows():
    print('Example Review:')
    print(row[0])
    print('Example Label:')
    print(row[1])
    break

""" Notice that the label is an integer. However, LLMs accept only strings. So we need to convert the integer label to a
 string label as we assemble the few-shot prompt. Let us assemble a few-shot prompt with 4 examples.
"""

for index, row in amazon_reviews_examples_df.sample(4).iterrows():
    example_review = row[0]
    example_label = row[1]

    few_shot_prompt.append(
        {
            'role': 'user',
            'content': example_review
        }
    )

    few_shot_prompt.append(
        {
            'role': 'assistant',
            'content': str(example_label) # LLMs accept only string inputs
        }
    )

few_shot_prompt

"""We now have 4 examples in the few shot prompt that is ready for use. Before we deploy this prompt, we need to get an estimate of the performance of this prompt. 
Here is where we use gold examples to estimate the accuracy.

## Evaluation
"""

predictions, ground_truths = [], []

for index, row in tqdm(amazon_reviews_gold_examples_df.iterrows()):
    gold_review = row[0]
    gold_label = row[1]

    user_input = [{'role':'user', 'content': gold_review}]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=few_shot_prompt + user_input,
            temperature=0
        )

        predictions.append(int(response.choices[0].message.content)) # convert the string label back to int
        ground_truths.append(gold_label)
    except Exception as e:
        print(e) # Log error and continue
        continue

predictions = np.array(predictions)
ground_truths = np.array(ground_truths)
(predictions == ground_truths).mean()

"""The output above indicates that the accuracy of the few-shot prompt on gold examples. More fine-grained evaluation 
(e.g., F1 score) could also be used to establish the estimated accuracy of the prompt.
"""

print(classification_report(ground_truths, predictions))

""">More examples does not imply better accuracy. Increasing the number of examples in the few-shot prompt beyond 16 is not 
known to yield better performance.

# Aspect-Based Sentiment Analysis (ABSA)

So far, we have been concerned with the overall sentiment of the input. However, when there are several aspects ("themes")
 that are included in the input text, it is not necessary that all the aspects of the input share the same sentiment. 
 For example, when reviewing a mobile phone, a customer might express sentiment about the different features of the phone 
 (e.g., camera, storage, screen, processor) and these features might not share the same sentiment. 
 It is likely that while the customer liked the camera, they expressed concern about the storage on the phone.

In aspect-based sentiment analysis (ABSA), the aim is to identify the aspects of entities and the sentiment expressed 
for each aspect. The ultimate goal is to be able to generate a measure of polarity that explicitly accounts for the 
different aspects of the input. In this way, ABSA presents a nuanced version of the overall polarity of the sentiment 
expresses in the input. For effective ABSA, we should be able to generate appropriate themes and then assign sentiment 
to the portions of the input that correspond to this theme.

## Prompt Design

We will design a few-shot prompt for ABSA. We first assemble a set of examples (input-output exemplars) as a part of the 
few-shot prompt. Then we create the prompt in the Anyscale API format.

To evaluate the performance of the *prompt* we will use a set of *gold examples*, that is, a set of examples that is 
not presented to the model as a part of the prompt.

**Examples and Gold Examples**

A set of examples and gold examples for aspect-based sentiment classification of restaurant reviews is hosted in a 
HuggingFace dataset. Let us load this data and take a look at the samples in this data.
"""

aspect_based_restaurant_reviews_corpus = load_dataset("jakartaresearch/semeval-absa", "restaurant")

restaurant_reviews_examples_df = aspect_based_restaurant_reviews_corpus['train'].to_pandas()
restaurant_reviews_gold_examples_df = aspect_based_restaurant_reviews_corpus['validation'].to_pandas()

examples_json = restaurant_reviews_examples_df.sample(4, random_state=42).to_json(orient='records')
gold_examples_json = restaurant_reviews_gold_examples_df.sample(32, random_state=42).to_json(orient='records')

json.loads(examples_json)

"""The labels we need are in the category key.

As the above code indicates, we have sampled 4 examples and 32 gold examples for ABSA.

**Assembling the prompt**
"""

few_shot_system_message = """
Perform aspect based sentiment analysis on restaurant reviews presented in the input delimited by triple backticks, that is, ```.
In each review there might be one or more of the following aspects: food, service, ambience, anecdotes/miscellaneous.
For each review presented as input:
- Identify if there are any of the 4 aspects (food, service, ambience, anecdotes/miscellaneous) present in the review.
- Assign a sentiment polarity (positive, negative or neutral) for each aspect

Arrange your response a JSON object with the following headers:
- category:[list of aspects]
- polarity:[list of corresponding polarities for each aspect]
"""

few_shot_prompt = [{'role':'system', 'content': few_shot_system_message}]

"""We need to iterate over the examples to append these examples as `user` and `assistant` messages to the few-shot prompt."""

for example in json.loads(examples_json):
    print(example['text'])
    print('--')
    print(example['category'])
    break

"""Notice that the label is an integer. However, LLMs accept only strings. So we need to convert the integer label to a 
string label as we assemble the few-shot prompt. Let us assemble a few-shot prompt with 4 examples.
"""

for example in json.loads(examples_json):
    example_review = example['text']
    example_label = example['category']

    few_shot_prompt.append(
        {
            'role': 'user',
            'content': example_review
        }
    )

    few_shot_prompt.append(
        {
            'role': 'assistant',
            'content': str(example_label)
        }
    )

few_shot_prompt

"""We now have 4 examples in the few shot prompt that is ready for use. Before we deploy this prompt, we need to get an 
estimate of the performance of this prompt. Here is where we use gold examples to estimate the accuracy.

## Evaluation

As in the case of sentiment analysis discussed in the previous task, we assign positive or negative sentiment to a review, 
with the additional objective of identifying entities (if any) present in the review.

To evaluate model performance, we judge the accuracy of the aspects + sentiment assignnment per aspect. Note that this is a 
much more stringent measure compared to the sentiment classification task we have seen so far. For example, if aspects identified by the LLM do not match the ground truth for a specific input, we count this prediction to be incorrect. A correct prediction is one where all the aspects are correctly idenfied and further the sentiment assignment for each aspect is also correctly identified (see figure below).
"""

model_predictions, ground_truths = [], []

for example in json.loads(gold_examples_json):
    gold_review = example['text']
    gold_label = example['category']

    user_input = [{'role':'user', 'content': gold_review}]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=few_shot_prompt + user_input,
            temperature=0
        )

        prediction = response.choices[0].message.content.replace("'", "\"")

        model_predictions.append(json.loads(prediction.strip().lower()))
        ground_truths.append(example['category'])

    except Exception as e:
        print(e) # Log error
        continue

def compute_accuracy(model_predictions, ground_truths, num_gold_examples=32):

    """
    Return the accuracy score comparing the model predictions and ground truth
    for ABSA. We look for exact matches between the model predictions on all the
    aspects and sentiments for these aspects in the ground truth.

    Args:
        gold_examples (str): JSON string with list of gold examples
        model_predictions (List): Nested list of ABSA predictions
        ground_truths (List): Nested list of ABSA annotations

    Output:
        accuracy (float): Exact matches of model predictions and ground truths
    """
    # Initialize variables to keep track of correct and total predictions
    correct_predictions = 0

    # Iterate through each prediction and ground truth pair
    for pred, truth in zip(model_predictions, ground_truths):
        if pred == truth:
            correct_predictions += 1

    # Calculate accuracy as the ratio of correct predictions to total predictions
    accuracy = correct_predictions / num_gold_examples

    return accuracy

compute_accuracy(model_predictions, ground_truths)