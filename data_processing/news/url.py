import time

from news.database import NewsDatabase
from news.engines.google import GoogleCSE
from news.engines.serp import SerpApi 
from news.engines.ddgs import DuckDuckGo 


class Scraper:

    def __init__(self, db, platform="serp") -> None:
        self.db: NewsDatabase = db
        self.scraped_urls = self.db.get_urls_as_dict()
        self.engine = self.get_engine(platform)

    def get_engine(self, platform):
        if platform.lower() == "serp":
            return SerpApi()
        elif platform.lower() == "ddgs":
            return DuckDuckGo(200)
        elif platform.lower() == "google":
            return GoogleCSE()


    def fill_news_urls(self, search_query, exact=False, publisher=None):
        query = search_query if not exact else f'"{search_query}"'

        if publisher:
            query += f' {publisher}'
            
        self.engine.set_search(query)


        while True:
            search_results = self.engine.next_page()
            # Check if there are still search results available
            if not search_results:
                break

            for result in search_results:
                news_url = result["link"]

                if news_url in self.scraped_urls:
                    continue

                self.db.add_entry(result['title'], url=news_url, category=search_query)
                self.scraped_urls[news_url] = result['title']
            time.sleep(5)
