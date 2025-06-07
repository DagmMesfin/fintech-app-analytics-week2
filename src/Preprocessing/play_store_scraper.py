from google_play_scraper import Sort, reviews
import csv
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(filename='../log/scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PlayStoreScraper:
    def __init__(self, app_name, app_id):
        self.app_name = app_name
        self.app_id = app_id

    def scrape_reviews(self):
        print(f"🔄 Fetching reviews for {self.app_name} ({self.app_id})...")
        logging.info("🔄 Fetching reviews...")

        try:
            results, _ = reviews(
                self.app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=500,
                filter_score_with=None
            )

            filename = f'../data/{self.app_name}_reviews.csv'

            logging.info(f"✅ Fetched {len(results)} reviews for {self.app_name} ({self.app_id})")

            # Write to CSV without using pandas
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['review', 'rating', 'date', 'app_name', 'source'])
                writer.writeheader()

                for entry in results:
                    writer.writerow({
                        'review': entry['content'],
                        'rating': entry['score'],
                        'date': entry['at'].strftime('%Y-%m-%d'),
                        'app_name': self.app_name,
                        'source': 'Google Play'
                    })

            logging.info(f"✅ Saved {len(results)} reviews to {filename}")
            print(f"✅ Saved {len(results)} reviews to {filename}")
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            print(f"❌ Error occurred: {e}")