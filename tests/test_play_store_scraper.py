import csv
import os
import builtins
from unittest.mock import patch

from src.Preprocessing.play_store_scraper import PlayStoreScraper


def test_scraper_writes_csv(tmp_path):
    from datetime import datetime

    # Prepare a fake reviews() response
    fake_results = [
        {
            "content": "Nice app",
            "score": 5,
            "at": datetime(2024, 1, 1),
        },
        {
            "content": "Crashes sometimes",
            "score": 2,
            "at": datetime(2024, 1, 2),
        },
    ]

    # Patch google_play_scraper.reviews to return our fake results
    with patch("src.Preprocessing.play_store_scraper.reviews", return_value=(fake_results, None)):
        scraper = PlayStoreScraper(app_name="CBE", app_id="com.example.cbe")
        # Create a layout where ../data from a working dir points to tmp/data
        data_dir = tmp_path / "data"
        work_dir = tmp_path / "work"
        data_dir.mkdir()
        work_dir.mkdir()
        with patch("builtins.open", wraps=builtins.open) as _:
            cwd = os.getcwd()
            try:
                os.chdir(work_dir)
                scraper.scrape_reviews()
            finally:
                os.chdir(cwd)

        # Verify file exists under ../data relative to work_dir -> tmp_path/data
        out_file = data_dir / "CBE_reviews.csv"
        assert out_file.exists()

        # Validate content
        with open(out_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["review"] == "Nice app"
        assert rows[1]["rating"] == "2"
