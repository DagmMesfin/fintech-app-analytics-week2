import pandas as pd
from src.Preprocessing.preprocessor import Preprocessor


def test_preprocessor_load_and_preprocess(sample_reviews_df, tmp_path):
    pp = Preprocessor([str(sample_reviews_df)])
    pp.load_data()

    # Ensure data loaded based on implementation printing (we just check internal state)
    assert len(pp.dataframes) == 1

    df = pp.preprocess_data()

    # Columns exist per implementation rename from app_name->bank
    assert set(["review", "rating", "date", "bank", "source"]).issubset(df.columns)

    # Dates parsed to datetime per implementation
    assert pd.api.types.is_datetime64_any_dtype(df["date"]) 

    # Saving works
    out = tmp_path / "out.csv"
    pp.save_data(str(out))
    loaded = pd.read_csv(out)
    assert len(loaded) == len(df)
