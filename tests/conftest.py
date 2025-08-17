import os
import pandas as pd
import pytest

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@pytest.fixture
def sample_reviews_df(tmp_path):
    df = pd.DataFrame({
        "review": ["Great app!", "Bad service.", "Okay experience"],
        "rating": [5, 1, 3],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "app_name": ["CBE", "Dashen", "BOA"],
        "source": ["Google Play", "Google Play", "Google Play"],
    })
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return p

@pytest.fixture
def thematic_results_df(tmp_path):
    df = pd.DataFrame({
        "review": ["Great app!", "Bad service."],
        "rating": [5, 1],
        "date": ["2024-01-01", "2024-01-02"],
        "bank": ["CBE", "Dashen"],
        "sentiment_label": ["POSITIVE", "NEGATIVE"],
        "sentiment_score": [0.98, 0.12],
        "themes": ["User Experience", "Issues / Pain Points"],
    })
    p = tmp_path / "thematic.csv"
    df.to_csv(p, index=False)
    return p
