
'''
import re
import spacy
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def preprocess_text(text):
    text = clean_text(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in STOP_WORDS and not token.is_punct]
    return " ".join(tokens)

def preprocess_dataframe(df):
    df["clean_review"] = df["review"].apply(preprocess_text)
    return df

# keyword_extraction.py
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(corpus, max_features=30, ngram_range=(2, 3)):
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english', max_features=max_features, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    keywords = vectorizer.get_feature_names_out()
    return keywords

def assign_themes(review, theme_dict):
    matched = []
    for theme, keywords in theme_dict.items():
        if any(kw in review for kw in keywords):
            matched.append(theme)
    return matched if matched else ["Uncategorized"]

THEME_KEYWORDS = {
    "User Experience": [
        "amazing", "easy", "fast", "good", "great", "nice", "super", "experience",
        "amazing app", "app easy", "app good", "easy use", "good app", "good application",
        "great app", "highly recommend", "nice app", "user friendly", "step ahead"
    ],
    "App Technology": [
        "app", "application", "mobile", "developer", "super app",
        "developer mode", "developer option", "digital banking", "mobile banking", "mobile banking app",
        "turn developer"
    ],
    "Bank Specific": [
        "bank", "banking", "cbe", "dashen", "dashen bank",
        "bank super", "bank super app", "dashen bank super", "dashen super", "dashen super app", "banking app", "super app", "supper app"
    ],
    "Issues / Pain Points": [
        "bad", "need", "work", "service", "slow", "crash", "freeze",
        "bad app", "need improvement", "app work", "transfer money"
    ],
    "Transaction & Performance": [
        "transaction", "transfer", "time", "payment", "fail", "transfer money"
    ],
    "Support": [
        "support", "help", "respond", "response", "call", "email", "feedback"
    ]
}

df = pd.read_csv("../data/bank_reviews_with_sentiment.csv")

# Preprocess reviews
print("Preprocessing reviews...")
df["clean_review"] = df["review"].astype(str).apply(preprocess_text)

# Extract global keywords (optional preview)
print("Extracting keywords...")
keywords = extract_keywords(df["clean_review"])
print(f"Top extracted keywords:\n{keywords}")

# Assign themes
print("Assigning themes...")
df["themes"] = df["clean_review"].apply(lambda text: assign_themes(text, THEME_KEYWORDS))

# Save results
df.to_csv("../data/thematic_results.csv", index=False)
print(f"Saved thematic results to: ../data/thematic_results.csv")

'''

# ThematicSentiment/ThematicAnalysis.py
import re
import spacy
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

class ThematicAnalysis:

    def __init__(self, theme_keywords=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.theme_keywords = theme_keywords if theme_keywords else {
            "User Experience": [
                "amazing", "easy", "fast", "good", "great", "nice", "super", "experience",
                "amazing app", "app easy", "app good", "easy use", "good app", "good application",
                "great app", "highly recommend", "nice app", "user friendly", "step ahead"
            ],
            "App Technology": [
                "app", "application", "mobile", "developer", "super app",
                "developer mode", "developer option", "digital banking", "mobile banking", 
                "mobile banking app", "turn developer"
            ],
            "Bank Specific": [
                "bank", "banking", "cbe", "dashen", "dashen bank",
                "bank super", "bank super app", "dashen bank super",
                "dashen super", "dashen super app", 
                "banking app",
                "super app", "supper app"
            ],
            "Issues / Pain Points": [
                "bad", "need", "work", "service", 
                "slow",  "crash", "freeze",
                "bad app", 
                "need improvement",
            ],
        }
        self.df = pd.read_csv("../data/bank_reviews_with_sentiment.csv")

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def preprocess_text(self, text):
        text = self.clean_text(text)
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in STOP_WORDS and not token.is_punct]
        return " ".join(tokens)
    
    def preprocess_dataframe(self):
        self.df["clean_review"] = self.df["review"].apply(self.preprocess_text)
        return self.df
    
    def extract_keywords(self, corpus, max_features=30, ngram_range=(2, 3)):
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english', 
                                     max_features=max_features, ngram_range=ngram_range)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        keywords = vectorizer.get_feature_names_out()
        return keywords
    
    def assign_themes(self, review):
        matched = []
        for theme, keywords in self.theme_keywords.items():
            if any(kw in review for kw in keywords):
                matched.append(theme)
        return matched if matched else ["Uncategorized"]
    
    def analyze_themes(self):
        # Preprocess reviews
        print("Preprocessing reviews...")
        df = self.preprocess_dataframe()

        # Extract global keywords (optional preview)
        print("Extracting keywords...")
        keywords = self.extract_keywords(self.df["clean_review"])
        print(f"Top extracted keywords:\n{keywords}")

        # Assign themes
        print("Assigning themes...")
        df["themes"] = self.df["clean_review"].apply(lambda text: self.assign_themes(text))

        return df
    
    def save_results(self, file_path="../data/thematic_results.csv"):
        self.df.to_csv(file_path, index=False)
        print(f"Saved thematic results to: {file_path}")