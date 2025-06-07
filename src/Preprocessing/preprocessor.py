import pandas as pd
from datetime import datetime

class Preprocessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.dataframes = []
        # Initialize an empty DataFrame for the final output with the expected structure
        self.final_df = pd.DataFrame(columns=['review', 'rating', 'date', 'app_name', 'source'])

    def load_data(self):
        for path in self.file_paths:
            try:
                df = pd.read_csv(path)
                self.dataframes.append(df)
                print(f"✅ Loaded data from {path}")
            except Exception as e:
                print(f"❌ Error loading data from {path}: {e}")

    def preprocess_data(self):
        for df in self.dataframes:
            # Remove duplicates
            df.drop_duplicates(subset=['review'], inplace=True)

            # Normalize dates
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Remove rows with NaN values in any column
            df.dropna(inplace=True)


        # Concatenate all dataframes
        self.final_df = pd.concat(self.dataframes, ignore_index=True)

        # Rename columns
        self.final_df.rename(columns={'app_name': 'bank'}, inplace=True)

        return self.final_df

    def save_data(self, output_path):
        try:
            self.final_df.to_csv(output_path, index=False)
            print(f"✅ Saved preprocessed data to {output_path}")
        except Exception as e:
            print(f"❌ Error saving preprocessed data: {e}")