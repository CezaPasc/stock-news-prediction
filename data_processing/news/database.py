import sqlite3
import csv

from news.entities import Article

class NewsDatabase:
    def __init__(self, db_file="news_db.sqlite"):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY,
                title TEXT,
                fullText TEXT,
                publisher TEXT,
                assets TEXT,
                date DATE,
                url TEXT UNIQUE,
                category TEXT,
                scraped INTEGER DEFAULT 0,
                error INTEGER DEFAULT 0,
                scrapedUrl TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def add_entry(self, title, url, category):
        try:
            self.cursor.execute("""
                INSERT INTO news (title, url, category)
                VALUES (?, ?, ?);
            """, (title, url, category))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print(f"URL already exists: {url}")

    def mark_error(self, url):
        query = """
            UPDATE news
            SET error = 1
            WHERE url = ?
        """
        self.cursor.execute(query, (url,))
        self.conn.commit()

        print(f"News entry for URL '{url}' updated successfully!")

    def get_latest_assets(self) -> list[str]:
        """
        Return all assets, where already in the last hour the URLs got scraped.
        This will be used to avoid searching for those assets again.
        """
        self.cursor.execute("SELECT category FROM news WHERE scrapedUrl >= DATE('now', '-1 hours') GROUP BY category;")
        return [row[0] for row in self.cursor.fetchall()]

    def update_news_entry(self, url, publisher, date, title, assets, full_text):
        """
        Update the specified fields for a specific URL in the 'news' table.

        Args:
            url (str): The unique URL of the news entry.
            publisher (str): The publisher name.
            date (str): The date in the format 'YYYY-MM-DD'.
            title (str): The news title.
            assets (list): A list of asset URLs.
            full_text (str): The full text of the news article.

        Returns:
            None
        """
        # Convert the list of assets to a comma-separated string
        assets_str = ', '.join(assets)

        # Execute the SQL update statement
        query = """
            UPDATE news
            SET publisher = ?,
                date = ?,
                title = ?,
                assets = ?,
                fullText = ?,
                scraped = 1
            WHERE url = ?
        """
        self.cursor.execute(query, (publisher, date, title, assets_str, full_text, url))
        self.conn.commit()

        print(f"News entry for URL '{url}' updated successfully!")

    def update_article(self, article: Article):
        self.update_news_entry(
            url=article.url,
            publisher=article.publisher_link,
            date=article.date,
            title=article.title,
            assets=article.assets,
            full_text=article.full_text
        )

    def get_all_urls(self):
        self.cursor.execute("SELECT url FROM news;")
        return [row[0] for row in self.cursor.fetchall()]

    def get_news_by_asset(self, asset):
        self.cursor.execute("""
            SELECT title, url FROM news
            WHERE ? IN (SELECT DISTINCT assets FROM news);
        """, (asset,))
        return self.cursor.fetchall()

    def export_to_csv(self, filename="news_data.csv"):
        self.cursor.execute("SELECT id, title, fullText, publisher, assets, date, url FROM news WHERE scraped = 1;")
        rows = self.cursor.fetchall()
        with open(filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["ID", "Title", "FullText", "Publisher", "tickers", "Date", "URL"])
            for row in rows:
                csv_writer.writerow(row)

    def get_urls_as_dict(self, non_scraped=False):
        query = "SELECT url, title FROM news WHERE scraped = 0 and error = 0;" if non_scraped else "SELECT url, title FROM news;"
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        return {url: title for url, title in rows}


    def close_connection(self):
        self.cursor.close()
        self.conn.close()


if __name__ == "__main__":
    db = NewsDatabase(db_file="test_news.sqlite")
    db.add_entry("This is just for testing", "https://yahoo.com/", "test")
    db.export_to_csv("scraped_yahoo_news.csv")