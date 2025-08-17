import pandas as pd
from src.Insight.Analysis.analyzer import InsightAnalyzer


def test_analyzer_outputs(tmp_path):
    df = pd.DataFrame({
        "review": ["Great", "Bad"],
        "rating": [5, 1],
        "date": ["2024-01-01", "2024-01-02"],
        "bank": ["CBE", "Dashen"],
        "sentiment_label": ["POSITIVE", "NEGATIVE"],
        "sentiment_score": [0.9, 0.1],
        "themes": ["User Experience", "Issues / Pain Points"],
    })
    p = tmp_path / "thematic.csv"
    df.to_csv(p, index=False)

    analyzer = InsightAnalyzer(thematic_results_path=str(p))
    pos, neg = analyzer.analyze_themes()

    assert not pos.empty
    assert not neg.empty

    summary = analyzer.summarize_sentiment()
    assert set(["bank", "sentiment_score", "rating"]).issubset(summary.columns)
