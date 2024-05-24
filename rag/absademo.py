from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline

# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained("deberta-v3-base-absa-v1.1", use_fast=False)
#absa_model = AutoModelForSequenceClassification \
#  .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

absa_model = AutoModelForSequenceClassification \
  .from_pretrained("deberta-v3-base-absa-v1.1")

# Load a traditional Sentiment Analysis model
#sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_model_path = "twitter-xlm-roberta-base-sentiment"
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path,
                          tokenizer=sentiment_model_path)

sentence = "We had a great experience at the restaurant, food was delicious, but " \
  "the service was kinda bad"
print(f"Sentence: {sentence}")
print()

# ABSA of "food"
aspect = "food"
inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
outputs = absa_model(**inputs)
probs = F.softmax(outputs.logits, dim=1)
probs = probs.detach().numpy()[0]
print(f"Sentiment of aspect '{aspect}' is:")
for prob, label in zip(probs, ["negative", "neutral", "positive"]):
  print(f"Label {label}: {prob}")
print()
# Sentiment of aspect 'food' is:
# Label negative: 0.0009989114478230476
# Label neutral: 0.001823813421651721
# Label positive: 0.997177243232727

# ABSA of "service"
aspect = "service"
inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
outputs = absa_model(**inputs)
probs = F.softmax(outputs.logits, dim=1)
probs = probs.detach().numpy()[0]
print(f"Sentiment of aspect '{aspect}' is:")
for prob, label in zip(probs, ["negative", "neutral", "positive"]):
  print(f"Label {label}: {prob}")
print()
# Sentiment of aspect 'service' is:
# Label negative: 0.9946129322052002
# Label neutral: 0.002369985682889819
# Label positive: 0.003017079783603549

# Overall sentiment of the sentence
sentiment = sentiment_model([sentence])[0]
print(f"Overall sentiment: {sentiment['label']} with score {sentiment['score']}")
# Overall sentiment: Negative with score 0.7706006765365601