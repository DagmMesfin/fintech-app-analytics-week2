import pandas as pd
from unittest.mock import patch
import spacy
from src.ThematicSentiment.ThematicAnalysis import ThematicAnalysis


def test_thematic_preprocess_and_assign(tmp_path):
    # Prepare a small CSV
    df = pd.DataFrame({
        "review": ["Amazing app, very easy to use", "Bad service, slow and crash"],
        "rating": [5, 1],
        "date": ["2024-01-01", "2024-01-02"],
        "bank": ["CBE", "Dashen"],
        "sentiment_label": ["POSITIVE", "NEGATIVE"],
        "sentiment_score": [0.95, 0.1],
    })
    p = tmp_path / "with_sentiment.csv"
    df.to_csv(p, index=False)

    # Use a blank English pipeline to avoid external model dependency in tests
    with patch("src.ThematicSentiment.ThematicAnalysis.spacy.load", side_effect=lambda name: spacy.blank("en")):
        ta = ThematicAnalysis()

    # Override dataframe with our test data
    ta.df = pd.read_csv(p)

    pre_df = ta.preprocess_dataframe()
    assert "clean_review" in pre_df.columns

    analyzed_df = ta.analyze_themes()
    assert "themes" in analyzed_df.columns
    # Should assign at least one theme per row or be non-null
    assert all((isinstance(x, list) and len(x) >= 1) for x in analyzed_df["themes"]) or analyzed_df["themes"].notna().all()
