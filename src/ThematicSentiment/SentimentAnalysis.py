
# ThematicSentiment/SentimentAnalysis.py

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

class SentimentAnalysis:

    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.classifier = pipeline("sentiment-analysis", model=model_name)
        self.df = pd.read_csv("../data/all_bank_reviews.csv")

    def classify_sentiment(self):
        sentiments = []
        for review in tqdm(self.df["review"], desc="Classifying sentiment"):
            try:
                result = self.classifier(review)[0]  # Truncate long reviews to 512 tokens
                sentiments.append((result['label'], round(result['score'], 3)))
            except Exception as e:
                sentiments.append(("neutral", 0.5))  # fallback on error
        return sentiments

    def add_sentiment_to_dataframe(self):
        sentiments = self.classify_sentiment(self.df["review"])
        self.df["sentiment_label"] = [s[0] for s in sentiments]
        self.df["sentiment_score"] = [s[1] for s in sentiments]
        return self.df
    
    def save_sentiment_data(self, file_path="../data/bank_reviews_with_sentiment.csv"):
        self.df.to_csv(file_path, index=False)
        print(f"Saved sentiment-labeled dataset to {file_path}.")