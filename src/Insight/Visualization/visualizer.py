'''
Visualizations
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class InsightVisualizer:
    def __init__(self, thematic_results_path="../data/thematic_results.csv"):
        self.df = pd.read_csv(thematic_results_path)

    def plot_sentiment_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='sentiment_label', order=self.df['sentiment_label'].value_counts().index)
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment Label")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    def plot_rating_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='bank', y='rating')
        plt.title("Rating Distribution by Bank")
        plt.xlabel("Bank")
        plt.ylabel("Rating")
        plt.xticks(rotation=45)
        plt.show()

    def generate_wordcloud(self):
        themes_text = ' '.join(self.df['themes'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(themes_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Theme Word Cloud")
        plt.show()