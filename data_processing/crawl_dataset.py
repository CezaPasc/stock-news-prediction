from tqdm import tqdm
import os
from datetime import datetime
import random

import reddit.data
from news import database
from news import url
import utils
import config
import twitter


if __name__ == "__main__":
    platform = "twitter"
    search_engine = "google" # google, ddgs or serp
    skip_url_crawling = True 
    parallel_scraping = True
    # Can be used to search for particularly publisher
    publisher = None # e.g: '"(Reuters)"'

    if platform == "reddit":
        reddit.data.crawl()
    elif platform == "twitter":
        twitter.crawl()
    elif platform == "news":
        db_file = os.path.join(config.data_dir, "yahoo", "big_news.sqlite")
        db = database.NewsDatabase(db_file=db_file)

        # Crawl URLs
        if not skip_url_crawling:
            url_scraper = url.Scraper(db, platform=search_engine)

            # Search for ticker names
            already_scraped = db.get_latest_assets()
            assets = [a for a in set(utils.get_asset_dict().values()) if a not in already_scraped]
            random.shuffle(assets)

            assets_tqdm = tqdm(assets)
            for asset in assets_tqdm:
                assets_tqdm.set_description("Searching Yahoo News URLs for '%s'" % asset)
                url_scraper.fill_news_urls(asset, publisher=publisher, exact=True)

        # Crawl articles
        news_urls = db.get_urls_as_dict(non_scraped=True)

        if parallel_scraping:
            # Parallel Process
            from news.scraper import ScrapeWorker, worker_task
            max_workers = 4
            results = []
            
            workers = [ScrapeWorker() for _ in range(max_workers)]

            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                print("Distributing tasks...")
                # Maps futures to their according urls
                future_to_url = {executor.submit(worker_task, workers[i % max_workers], url): url for i, url in enumerate(news_urls)}
                
                for future in tqdm(as_completed(future_to_url), total=len(news_urls)):
                    # Look up which URL this job belongs to
                    url = future_to_url[future]
                    try:
                        article = future.result()
                        if article:
                            db.update_article(article)
                    except Exception as e:
                        print(f"Error scraping {url}: {e}")        
                        db.mark_error(url)
            [worker.close() for worker in workers]
        else:
            # Sequential Process
            from news.scraper import ScrapeWorker
            worker = ScrapeWorker()
            for url in tqdm(news_urls.keys()):
                try:
                    article = worker.scrape(url)
                    db.update_article(article)
                except Exception as e:
                    print("Error while scraping")
                    print(e)
                    db.mark_error(url)
            worker.close()

        output_name = "news_export_%s.csv" % datetime.now().strftime("%Y-%m-%d")
        db.export_to_csv(output_name)
        print("Exported news article to %s" % output_name)

        