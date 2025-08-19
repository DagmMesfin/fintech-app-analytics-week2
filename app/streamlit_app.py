import os
import re
from statistics import mean
from typing import List, Dict

import pandas as pd
import streamlit as st
import psycopg2
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from google_play_scraper import reviews, Sort

# ------------------------------
# Caching heavy resources
# ------------------------------
@st.cache_resource(show_spinner=False)
def get_sentiment_classifier():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# ------------------------------
# Preprocessing
# ------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize columns if possible
    rename_map = {"app_name": "bank"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Ensure required columns
    for col in ["review", "rating", "date"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Deduplicate on review text
    df.drop_duplicates(subset=["review"], inplace=True)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with any NaNs
    df.dropna(inplace=True)

    return df


# ------------------------------
# Sentiment
# ------------------------------
def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    clf = get_sentiment_classifier()
    labels = []
    scores = []

    progress = st.progress(0)
    total = len(df)
    for i, text in enumerate(df["review"].astype(str).tolist(), start=1):
        try:
            res = clf(text)[0]
            labels.append(res.get("label", "NEUTRAL"))
            scores.append(float(res.get("score", 0.0)))
        except Exception:
            labels.append("NEUTRAL")
            scores.append(0.0)
        if i % 10 == 0 or i == total:
            progress.progress(i / total)

    df = df.copy()
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    return df


# ------------------------------
# Thematic (keyword-based)
# ------------------------------
THEME_KEYWORDS: Dict[str, List[str]] = {
    "User Experience": [
        "amazing", "easy", "fast", "good", "great", "nice", "super", "experience",
        "amazing app", "app easy", "app good", "easy use", "good app", "good application",
        "great app", "highly recommend", "nice app", "user friendly", "step ahead",
    ],
    "App Technology": [
        "app", "application", "mobile", "developer", "super app",
        "developer mode", "developer option", "digital banking", "mobile banking", "mobile banking app",
        "turn developer",
    ],
    "Bank Specific": [
        "bank", "banking", "cbe", "dashen", "dashen bank", "boa",
        "bank super", "bank super app", "dashen bank super", "dashen super", "dashen super app", "banking app", "super app", "supper app",
    ],
    "Issues / Pain Points": [
        "bad", "need", "work", "service", "slow", "crash", "freeze", "issue", "problem",
        "bad app", "need improvement", "app work", "transfer money",
    ],
}


def _clean_text_basic(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def assign_themes_to_row(review: str) -> List[str]:
    t = _clean_text_basic(review)
    matched: List[str] = []
    for theme, kws in THEME_KEYWORDS.items():
        if any(kw in t for kw in kws):
            matched.append(theme)
    return matched if matched else ["Uncategorized"]


def run_thematic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_review"] = df["review"].astype(str).apply(_clean_text_basic)
    df["themes"] = df["clean_review"].apply(assign_themes_to_row)
    return df


# ------------------------------
# Aspect-Based Sentiment Analysis (ABSA)
# ------------------------------
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "login": ["login", "log in", "signin", "sign in", "password", "otp"],
    "speed": ["slow", "lag", "fast", "quick", "speed", "responsive", "delay"],
    "reliability": ["reliable", "unreliable", "downtime", "outage", "server down", "stable", "unstable"],
    "performance": ["performance", "crash", "freeze", "hang", "bug", "buggy", "glitch"],
    "ui": ["ui", "ux", "interface", "design", "layout", "user friendly", "navigation"],
    "payment": ["payment", "pay", "bill", "merchant", "qr", "pos"],
    "transfer": ["transfer", "send", "receive", "transaction", "funds", "remit"],
    "security": ["secure", "security", "safe", "fraud", "scam", "pin", "biometric"],
    "notification": ["notification", "alert", "reminder"],
    "update": ["update", "version", "upgrade"],
    "install": ["install", "installation", "download"],
    "account": ["account", "balance", "statement", "history", "limit"],
    "customer_support": ["support", "help", "service", "response", "call", "email", "feedback"],
}


def _sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", str(text)) if s.strip()]


def _normalize(text: str) -> str:
    return str(text).lower()


def extract_aspects(text: str, aspect_keywords: Dict[str, List[str]]) -> List[str]:
    t = _normalize(text)
    found: List[str] = []
    for aspect, kws in aspect_keywords.items():
        for kw in kws:
            if kw in t:
                found.append(aspect)
                break
    return sorted(set(found))


def _score_text_sentiment(text: str) -> Dict[str, float]:
    clf = get_sentiment_classifier()
    try:
        res = clf(str(text))[0]
        return {"label": res.get("label", "NEUTRAL"), "score": float(res.get("score", 0.0))}
    except Exception:
        return {"label": "NEUTRAL", "score": 0.0}


def aspect_sentiment_for_review(text: str, aspects: List[str], aspect_keywords: Dict[str, List[str]], neutral_threshold: float = 0.0) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    sentences = _sentence_split(text)
    for aspect in aspects:
        kws = aspect_keywords.get(aspect, [])
        target_sentences = [s for s in sentences if any(kw in _normalize(s) for kw in kws)]
        if not target_sentences:
            target_sentences = [text]
        scored = [_score_text_sentiment(s) for s in target_sentences]
        pos_scores = [s["score"] for s in scored if s["label"].upper().startswith("POS")]
        neg_scores = [s["score"] for s in scored if s["label"].upper().startswith("NEG")]
        if not pos_scores and not neg_scores:
            results[aspect] = {"label": "NEUTRAL", "score": 0.0}
            continue
        avg_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
        avg_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
        if max(avg_pos, avg_neg) < neutral_threshold:
            results[aspect] = {"label": "NEUTRAL", "score": max(avg_pos, avg_neg)}
        else:
            label = "POSITIVE" if avg_pos >= avg_neg else "NEGATIVE"
            results[aspect] = {"label": label, "score": max(avg_pos, avg_neg)}
    return results


def run_absa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["aspects"] = df["review"].apply(lambda x: extract_aspects(x, ASPECT_KEYWORDS))
    df["aspect_sentiments"] = df.apply(
        lambda r: aspect_sentiment_for_review(r["review"], r["aspects"], ASPECT_KEYWORDS) if r["aspects"] else {},
        axis=1,
    )
    # build long-form
    rows: List[Dict] = []
    for idx, r in df.iterrows():
        bank = r["bank"] if "bank" in df.columns else None
        rating = r["rating"] if "rating" in df.columns else None
        for aspect in r["aspects"]:
            s = r["aspect_sentiments"].get(aspect, {"label": "NEUTRAL", "score": 0.0})
            rows.append({
                "review_id": idx,
                "bank": bank,
                "aspect": aspect,
                "aspect_label": s["label"],
                "aspect_score": s["score"],
                "rating": rating,
            })
    return pd.DataFrame(rows)


# ------------------------------
# Database (PostgreSQL)
# ------------------------------

def get_pg_connection():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "fintech"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "postgres"),
    )


def save_dataframe_to_postgres(df: pd.DataFrame):
    conn = get_pg_connection()
    cur = conn.cursor()

    # Create tables
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS banks (
            bank_id INTEGER PRIMARY KEY,
            bank_name VARCHAR(100)
        );

        CREATE TABLE IF NOT EXISTS reviews (
            review_id VARCHAR(50) PRIMARY KEY,
            review_text TEXT,
            rating INTEGER,
            review_date DATE,
            sentiment_label VARCHAR(20),
            sentiment_score DOUBLE PRECISION,
            themes VARCHAR(100),
            bank_id INTEGER REFERENCES banks(bank_id)
        );
        """
    )

    # Build bank mapping
    banks = sorted([b for b in df["bank"].dropna().unique().tolist()]) if "bank" in df.columns else []
    bank_map = {b: i + 1 for i, b in enumerate(banks)}

    # Insert banks
    if banks:
        cur.executemany(
            "INSERT INTO banks (bank_id, bank_name) VALUES (%s, %s) ON CONFLICT (bank_id) DO NOTHING",
            [(bank_map[b], b) for b in banks],
        )

    # Prepare reviews
    df = df.copy()
    if "bank" in df.columns:
        df["bank_id"] = df["bank"].map(bank_map)
    df["review_id"] = df.index.astype(str)

    rows = df[[
        "review_id",
        "review",
        "rating",
        "date",
        "sentiment_label",
        "sentiment_score",
        "themes",
        "bank_id",
    ]].copy()

    # Flatten list themes to comma-separated strings
    rows["themes"] = rows["themes"].apply(lambda x: ", ".join(x) if isinstance(x, list) else (x or ""))

    cur.executemany(
        """
        INSERT INTO reviews (
            review_id, review_text, rating, review_date,
            sentiment_label, sentiment_score, themes, bank_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (review_id) DO NOTHING
        """,
        rows.values.tolist(),
    )

    conn.commit()
    cur.close()
    conn.close()


# ------------------------------
# Scraping (Google Play)
# ------------------------------
@st.cache_data(show_spinner=False)
def scrape_play_store(app_id: str, app_name: str, count: int = 200, lang: str = "en", country: str = "us") -> pd.DataFrame:
    """Return a DataFrame with columns: review, rating, date, bank, source."""
    result, _ = reviews(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=count,
        filter_score_with=None,
    )
    rows = []
    for r in result:
        rows.append({
            "review": r.get("content", ""),
            "rating": r.get("score", None),
            "date": r.get("at").strftime("%Y-%m-%d") if r.get("at") else None,
            "bank": app_name,
            "source": "Google Play",
        })
    return pd.DataFrame(rows)


def parse_bank_appid_lines(text: str) -> List[Dict[str, str]]:
    """Parse lines in the form 'BankName, app_id' into [{bank, app_id}, ...]."""
    pairs: List[Dict[str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "," in line:
            parts = [p.strip() for p in line.split(",", 1)]
        else:
            parts = line.split()
        if len(parts) >= 2:
            pairs.append({"bank": parts[0], "app_id": parts[1]})
    return pairs


# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Fintech App Reviews - Analytics", layout="wide")
st.title("Customer Experience Analytics for Fintech Apps")

with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Choose input", ["Scrape Google Play", "Upload CSV", "Existing CSV in data/"])
    uploaded_files = None
    if mode == "Upload CSV":
        uploaded_files = st.file_uploader(
            "Upload CSV file(s) with columns: review, rating, date, app_name/bank, source",
            type=["csv"],
            accept_multiple_files=True,
        )
    elif mode == "Scrape Google Play":
        st.caption("Enter one per line: BankName, app_id")
        st.caption("Example: CBE, com.example.cbe")
        entries_text = st.text_area("Apps to scrape", value="", height=120)
        col_scr1, col_scr2 = st.columns(2)
        with col_scr1:
            scrape_count = st.number_input("Reviews per app", min_value=50, max_value=2000, value=500, step=50)
            scrape_lang = st.text_input("Language", value="en")
        with col_scr2:
            scrape_country = st.text_input("Country", value="us")
            if st.button("Fetch reviews"):
                pairs = parse_bank_appid_lines(entries_text)
                if not pairs:
                    st.error("Provide at least one 'BankName, app_id' entry.")
                else:
                    scraped_dfs = []
                    for p in pairs:
                        try:
                            df_scr = scrape_play_store(p["app_id"], p["bank"], count=int(scrape_count), lang=scrape_lang, country=scrape_country)
                            scraped_dfs.append(df_scr)
                        except Exception as e:
                            st.error(f"Failed to scrape {p['bank']} ({p['app_id']}): {e}")
                    if scraped_dfs:
                        st.session_state.scraped_df = pd.concat(scraped_dfs, ignore_index=True)
                        st.success(f"Scraped {len(st.session_state.scraped_df)} reviews.")
    else:
        st.info("Using existing CSVs from data/ folder. Adjust below if needed.")

    st.header("Database")
    st.text_input("PGHOST", value=os.getenv("PGHOST", "localhost"), key="PGHOST")
    st.text_input("PGPORT", value=os.getenv("PGPORT", "5432"), key="PGPORT")
    st.text_input("PGDATABASE", value=os.getenv("PGDATABASE", "fintech"), key="PGDATABASE")
    st.text_input("PGUSER", value=os.getenv("PGUSER", "postgres"), key="PGUSER")
    st.text_input("PGPASSWORD", value=os.getenv("PGPASSWORD", "postgres"), key="PGPASSWORD", type="password")
    if st.button("Apply DB Env"):
        os.environ.update({
            "PGHOST": st.session_state.PGHOST,
            "PGPORT": st.session_state.PGPORT,
            "PGDATABASE": st.session_state.PGDATABASE,
            "PGUSER": st.session_state.PGUSER,
            "PGPASSWORD": st.session_state.PGPASSWORD,
        })
        st.success("Database environment variables applied.")

# Load data
input_dfs: List[pd.DataFrame] = []
if mode == "Upload CSV" and uploaded_files:
    for f in uploaded_files:
        try:
            input_dfs.append(pd.read_csv(f))
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
elif mode == "Existing CSV in data/":
    data_dir = os.path.join(os.getcwd(), "data")
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            if name.lower().endswith(".csv"):
                try:
                    input_dfs.append(pd.read_csv(os.path.join(data_dir, name)))
                except Exception:
                    pass
elif mode == "Scrape Google Play":
    if "scraped_df" in st.session_state:
        input_dfs = [st.session_state.scraped_df]
    else:
        st.info("Provide apps and click 'Fetch reviews' in the sidebar, then run again.")

if not input_dfs:
    st.warning("No input data found. Upload CSVs, scrape from Google Play, or place CSVs in data/.")
    st.stop()

raw_df = pd.concat(input_dfs, ignore_index=True)
st.subheader("Raw Data Sample")
st.dataframe(raw_df.head(10), use_container_width=True)

# Run pipeline
st.header("Pipeline")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("1) Preprocess"):
        try:
            st.session_state.preprocessed = preprocess_dataframe(raw_df)
            st.success("Preprocessing complete")
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")

with col2:
    if st.button("2) Sentiment"):
        if "preprocessed" not in st.session_state:
            st.error("Run Preprocess first")
        else:
            with st.spinner("Running sentiment analysis..."):
                st.session_state.with_sentiment = run_sentiment(st.session_state.preprocessed)
            st.success("Sentiment complete")

with col3:
    if st.button("3) Thematic"):
        target_key = "with_sentiment" if "with_sentiment" in st.session_state else "preprocessed"
        if target_key not in st.session_state:
            st.error("Run Preprocess (and optionally Sentiment) first")
        else:
            with st.spinner("Assigning themes..."):
                st.session_state.thematic = run_thematic(st.session_state[target_key])
            st.success("Thematic complete")

with col4:
    if st.button("4) Save to PostgreSQL"):
        if "thematic" not in st.session_state:
            st.error("Run Thematic first")
        else:
            try:
                save_dataframe_to_postgres(st.session_state.thematic)
                st.success("Saved to PostgreSQL")
            except Exception as e:
                st.error(f"DB save failed: {e}")

# Show outputs
if "with_sentiment" in st.session_state:
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="sentiment_label", data=st.session_state.with_sentiment, ax=ax,
                  order=st.session_state.with_sentiment["sentiment_label"].value_counts().index)
    st.pyplot(fig)

if "thematic" in st.session_state:
    st.subheader("Themes per Bank (Top 10)")
    df_themes = st.session_state.thematic.copy()
    # explode list of themes
    exploded = df_themes.explode("themes")
    if "bank" in exploded.columns:
        top = (exploded.groupby(["bank", "themes"]).size().reset_index(name="count")
               .sort_values(["bank", "count"], ascending=[True, False]).groupby("bank").head(10))
        st.dataframe(top, use_container_width=True)

    st.subheader("Download Processed CSV")
    st.download_button(
        label="Download thematic_results.csv",
        data=st.session_state.thematic.to_csv(index=False),
        file_name="thematic_results.csv",
        mime="text/csv",
    )

# ABSA Section
st.header("Aspect-Based Sentiment Analysis (ABSA)")
if st.button("Run ABSA"):
    target_key = None
    # Prefer thematic output, else with_sentiment, else preprocessed
    for key in ["thematic", "with_sentiment", "preprocessed"]:
        if key in st.session_state:
            target_key = key
            break
    if not target_key:
        st.error("Run at least Preprocess first")
    else:
        with st.spinner("Running ABSA..."):
            st.session_state.absa_long = run_absa(st.session_state[target_key])
        if st.session_state.absa_long.empty:
            st.warning("No aspects detected. Consider expanding aspect keywords.")
        else:
            st.success("ABSA complete")

if "absa_long" in st.session_state and not st.session_state.absa_long.empty:
    st.subheader("ABSA Sample")
    st.dataframe(st.session_state.absa_long.head(20), use_container_width=True)

    st.subheader("Counts by Aspect/Label")
    agg_counts = (st.session_state.absa_long.groupby(["aspect", "aspect_label"]).size().reset_index(name="count"))
    st.dataframe(agg_counts.sort_values(["aspect", "count"], ascending=[True, False]).head(50), use_container_width=True)

    st.subheader("Average Aspect Score")
    agg_scores = (st.session_state.absa_long.groupby(["aspect"])["aspect_score"].mean().reset_index(name="avg_aspect_score"))
    st.dataframe(agg_scores.sort_values("avg_aspect_score", ascending=False), use_container_width=True)
