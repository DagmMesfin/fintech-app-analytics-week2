
import pandas as pd
import oracledb

def save_to_oracle():
    # Load your cleaned DataFrame
    df = pd.read_csv("../data/thematic_results.csv")
    bank_id_map = {"CBE": 1, "Dashen": 2, "BOA": 3}
    df["bank_id"] = df["bank"].map(bank_id_map)
    df.rename(columns={"date": "review_date"}, inplace=True)
    df.rename(columns={"review": "review_text"}, inplace=True)

    df["review_id"] = df.index.astype(str)

    # Connect to Oracle DB
    conn = oracledb.connect(user="SYSTEM", password="12121212", dsn="localhost/XEPDB1")
    cursor = conn.cursor()

    #clear the tables if they exist
    cursor.execute("TRUNCATE TABLE reviews")
    cursor.execute("TRUNCATE TABLE banks")

    # Create tables (if not already created)
    cursor.execute("""
    BEGIN
    EXECUTE IMMEDIATE 'CREATE TABLE banks (
        bank_id NUMBER PRIMARY KEY,
        bank_name VARCHAR2(100)
    )';
    EXCEPTION
    WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF;
    END;
    """)

    cursor.execute("""
    BEGIN
    EXECUTE IMMEDIATE 'CREATE TABLE reviews (
        review_id VARCHAR2(50) PRIMARY KEY,
        review_text CLOB,
        rating NUMBER,
        review_date DATE,
        sentiment_label VARCHAR2(20),
        sentiment_score FLOAT,
        themes VARCHAR2(100),
        bank_id NUMBER,
        FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
    )';
    EXCEPTION
    WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF;
    END;
    """)

    # Insert into banks
    banks = df[['bank_id', 'bank']].drop_duplicates().values.tolist()
    cursor.executemany("INSERT INTO banks (bank_id, bank_name) VALUES (:1, :2)", banks)

    # Insert into reviews
    data = df[[
        "review_id", "review_text", "rating", "review_date",
        "sentiment_label", "sentiment_score", "themes", "bank_id"
    ]].values.tolist()

    cursor.executemany("""
    INSERT INTO reviews (
        review_id, review_text, rating, review_date,
        sentiment_label, sentiment_score, themes, bank_id
    )
    VALUES (
        :1, :2, :3, TO_DATE(:4, 'YYYY-MM-DD'),
        :5, :6, :7, :8
    )
    """, data)

    # Commit and close
    conn.commit()
    cursor.close()
    conn.close()

    print("✅ Data inserted using oracledb!")