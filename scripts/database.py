import os
import pandas as pd
import psycopg2


def get_pg_connection():
    """Create a PostgreSQL connection using environment variables with sensible defaults.
    Env vars: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
    """
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "fintech"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "postgres"),
    )


def save_to_postgres():
    # Load your cleaned DataFrame
    df = pd.read_csv("../data/thematic_results.csv")
    bank_id_map = {"CBE": 1, "Dashen": 2, "BOA": 3}
    df["bank_id"] = df["bank"].map(bank_id_map)
    df.rename(columns={"date": "review_date"}, inplace=True)
    df.rename(columns={"review": "review_text"}, inplace=True)

    df["review_id"] = df.index.astype(str)

    # Connect to PostgreSQL
    conn = get_pg_connection()
    cursor = conn.cursor()

    # Recreate tables to ensure schema matches Postgres
    cursor.execute(
        """
        DROP TABLE IF EXISTS reviews;
        DROP TABLE IF EXISTS banks;

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

    # Insert into banks
    banks = df[["bank_id", "bank"]].drop_duplicates().values.tolist()
    cursor.executemany(
        "INSERT INTO banks (bank_id, bank_name) VALUES (%s, %s)",
        banks,
    )

    # Insert into reviews
    data = df[ [
        "review_id",
        "review_text",
        "rating",
        "review_date",
        "sentiment_label",
        "sentiment_score",
        "themes",
        "bank_id",
    ]].values.tolist()

    cursor.executemany(
        """
        INSERT INTO reviews (
            review_id, review_text, rating, review_date,
            sentiment_label, sentiment_score, themes, bank_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        data,
    )

    # Commit and close
    conn.commit()
    cursor.close()
    conn.close()

    print("✅ Data inserted using PostgreSQL!")