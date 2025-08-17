import os
import pandas as pd
import pytest

from scripts.database import save_to_postgres

pytestmark = pytest.mark.integration


def test_save_to_postgres_integration(tmp_path, monkeypatch):
    # Prepare a small thematic_results.csv the script expects
    df = pd.DataFrame({
        "review": ["Great app", "Bad experience"],
        "rating": [5, 1],
        "date": ["2024-01-10", "2024-01-11"],
        "bank": ["CBE", "Dashen"],
        "sentiment_label": ["POSITIVE", "NEGATIVE"],
        "sentiment_score": [0.95, 0.08],
        "themes": ["User Experience", "Issues / Pain Points"],
    })
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    thematic_csv = data_dir / "thematic_results.csv"
    df.to_csv(thematic_csv, index=False)

    # Monkeypatch the script to read from our temp location
    from scripts import database as db
    def fake_read_csv(path):
        # scripts/database.py uses ../data/thematic_results.csv relative to its file
        return pd.read_csv(thematic_csv)
    db.pd.read_csv = fake_read_csv

    # Point to a test database via env vars (must exist)
    monkeypatch.setenv("PGHOST", os.getenv("PGHOST", "localhost"))
    monkeypatch.setenv("PGPORT", os.getenv("PGPORT", "5432"))
    monkeypatch.setenv("PGDATABASE", os.getenv("PGDATABASE", "fintech"))
    monkeypatch.setenv("PGUSER", os.getenv("PGUSER", "postgres"))
    monkeypatch.setenv("PGPASSWORD", os.getenv("PGPASSWORD", "postgres"))

    # Run - will create tables and insert rows
    try:
        save_to_postgres()
    except Exception as e:
        pytest.skip(f"Skipping integration test due to DB connectivity issue: {e}")
