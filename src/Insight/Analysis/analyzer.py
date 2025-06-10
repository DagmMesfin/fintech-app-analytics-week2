
'''
InsightAnalyzer: A class to analyze thematic results from bank reviews.

'''

import pandas as pd
import ast

class InsightAnalyzer:
    def __init__(self, thematic_results_path = "../data/thematic_results.csv"):
        self.df = pd.read_csv(thematic_results_path)

    def analyze_themes(self):
        # Filter positive reviews
        positive_reviews = self.df[self.df['sentiment_label'] == 'POSITIVE']

        # Filter negative reviews
        negative_reviews = self.df[self.df['sentiment_label'] == 'NEGATIVE']

        # Get top positive themes per bank
        top_positive_themes = positive_reviews.groupby(['bank', 'themes']).size().reset_index(name='count')
        top_positive_themes = top_positive_themes.sort_values(['bank', 'count'], ascending=[True, False])

        # Get top negative themes per bank
        top_negative_themes = negative_reviews.groupby(['bank', 'themes']).size().reset_index(name='count')
        top_negative_themes = top_negative_themes.sort_values(['bank', 'count'], ascending=[True, False])

        print("Top Positive Themes (Drivers) per Bank:")
        print(top_positive_themes.groupby('bank').head(3))

        print("\nTop Negative Themes (Pain Points) per Bank:")
        print(top_negative_themes.groupby('bank').head(3))

        return top_positive_themes, top_negative_themes

    def summarize_sentiment(self):
        summary = self.df.groupby('bank').agg({
            'sentiment_score': 'mean',
            'rating': 'mean'
        }).reset_index()

        return summary