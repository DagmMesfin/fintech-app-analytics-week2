# Customer Experience Analytics for Fintech Apps

Analyze Google Play Store reviews for fintech banking apps to extract sentiment and themes, generate insights, and persist results to a PostgreSQL database.

## Key Features
- Automated scraping of Play Store reviews
- Text preprocessing (cleaning, normalization)
- Sentiment analysis (Transformer/ML-based)
- Thematic/topic analysis
- Insight generation and visualization
- Persistence to PostgreSQL (banks and reviews tables)
- Jupyter notebooks for exploration and reporting

## Project Structure
```
fintech-app-analytics-week2/
├─ app/
├─ data/
│  ├─ all_bank_reviews.csv
│  ├─ bank_reviews_with_sentiment.csv
│  ├─ BOA_reviews.csv
│  ├─ CBE_reviews.csv
│  ├─ Dashen_reviews.csv
│  └─ thematic_results.csv   # final processed dataset inserted to DB
├─ notebooks/
│  ├─ scraping_and_preprocessing.ipynb
│  ├─ sentiment_and_thematic_analysis.ipynb
│  ├─ insight_and_recommendation.ipynb
│  └─ save_to_database.ipynb
├─ scripts/
│  ├─ database.py            # save_to_postgres()
│  └─ README.md
├─ src/
│  ├─ Preprocessing/
│  │  ├─ play_store_scraper.py
│  │  └─ preprocessor.py
│  ├─ ThematicSentiment/
│  │  ├─ SentimentAnalysis.py
│  │  └─ ThematicAnalysis.py
│  └─ Insight/
│     ├─ Analysis/analyzer.py
│     └─ Visualization/visualizer.py
├─ tests/
├─ requirements.txt
└─ README.md
```

## Data Pipeline (High Level)
1. Scrape reviews per bank from Google Play (`src/Preprocessing/play_store_scraper.py`).
2. Clean and normalize text (`src/Preprocessing/preprocessor.py`).
3. Run sentiment analysis (`src/ThematicSentiment/SentimentAnalysis.py`).
4. Run thematic/topic analysis (`src/ThematicSentiment/ThematicAnalysis.py`).
5. Aggregate results to `data/thematic_results.csv`.
6. Persist to PostgreSQL using `scripts/database.py`.
7. Explore insights and visuals in notebooks under `notebooks/`.

## Requirements
- Python 3.10+ recommended
- PostgreSQL 13+ (local or remote)
- Packages listed in `requirements.txt` (key: pandas, transformers, torch, scikit-learn, spacy, seaborn, matplotlib, psycopg2-binary, ipykernel)

Optional (if using spaCy):
```
python -m spacy download en_core_web_sm
```

## Setup (Windows PowerShell)
1. Create and activate a virtual environment
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Configure PostgreSQL
1. Ensure a PostgreSQL server is running and that you have a database (default used below is `fintech`).
2. Set environment variables before running the scripts:
```
$env:PGHOST = "localhost"
$env:PGPORT = "5432"
$env:PGDATABASE = "fintech"
$env:PGUSER = "postgres"
$env:PGPASSWORD = "postgres"
```
You can persist these via a profile or manage them with a `.env` loader if desired.

## Usage
- Run the notebooks in `notebooks/` in order for end-to-end exploration:
  - `scraping_and_preprocessing.ipynb`
  - `sentiment_and_thematic_analysis.ipynb`
  - `insight_and_recommendation.ipynb`
  - `save_to_database.ipynb` (optional; or call the script below)

- Programmatic insert to PostgreSQL from the final CSV (`data/thematic_results.csv`):
```
python -c "from scripts.database import save_to_postgres; save_to_postgres()"
```
This will:
- Create tables if they do not exist
- Insert unique banks into `banks`
- Insert reviews and associated metadata into `reviews`

### Database Schema
- banks
  - bank_id INTEGER PRIMARY KEY
  - bank_name VARCHAR(100)
- reviews
  - review_id VARCHAR(50) PRIMARY KEY
  - review_text TEXT
  - rating INTEGER
  - review_date DATE
  - sentiment_label VARCHAR(20)
  - sentiment_score DOUBLE PRECISION
  - themes VARCHAR(100)
  - bank_id INTEGER REFERENCES banks(bank_id)

Note: The script drops and recreates tables by default for a clean load. Remove the drops if you need to append data.

## Development Notes
- Code lives under `src/` organized by concern (Preprocessing, ThematicSentiment, Insight).
- Visualizations and analysis helpers in `src/Insight/Visualization/visualizer.py` and `src/Insight/Analysis/analyzer.py`.
- Data artifacts live in `data/`. Generated files are ignored from VCS as appropriate.

## Testing
- Basic test scaffolding under `tests/`. Extend with unit tests for preprocessing, sentiment, and DB logic as needed (pytest recommended).

## Troubleshooting
- psycopg2 install: use `psycopg2-binary` (already in requirements). For production, prefer building `psycopg2` from source.
- SSL/connection issues: verify the PG env vars and that the database is reachable.
- Date parsing: `review_date` should be ISO-like (YYYY-MM-DD). Adjust parsing in `scripts/database.py` if your CSV differs.
- Large CSVs: consider chunked inserts or COPY for performance.

## License
MIT License