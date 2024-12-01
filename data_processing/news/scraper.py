from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from urllib.parse import urlparse, urlunparse

import time
import random
from news.entities import Article



def transform_url(url):
    parsed_url = urlparse(url)
    # Remove the country code from the netloc
    netloc_parts = parsed_url.netloc.split('.')
    if len(netloc_parts) > 2 and len(netloc_parts[0]) == 2:
        netloc_parts.pop(0)
    new_netloc = '.'.join(netloc_parts)
    
    # Reconstruct the URL with the modified netloc
    new_url = urlunparse(parsed_url._replace(netloc=new_netloc))
    return new_url

def sleep(mean, std_dev):
    # Generate a random sleep duration based on the normal distribution
    sleep_duration = random.gauss(mean, std_dev)
    # Ensure the sleep duration is non-negative
    sleep_duration = max(0, sleep_duration)
    # Sleep for the calculated duration
    time.sleep(sleep_duration)


class ScrapeWorker:
    def __init__(self) -> None:
        self.driver = webdriver.Firefox()
        print("ScrapeWorker initialized")

    def click_if_exists(self, class_name):
        print(f"Looking for a {class_name} button")
        button = self.driver.find_elements(By.CLASS_NAME, class_name)
        if button:
            print("Clicking this button")
            # TODO: Check if delay is necessary
            sleep(0.2, 0.02)
            button[0].send_keys(Keys.ENTER)
            sleep(1, 0.02)
        
    def scrape(self, url):
        sanitized_url = transform_url(url)
        # Navigate to the desired webpage
        print(f"Opening {sanitized_url}")
        self.driver.get(sanitized_url)
        sleep(0.2, 0.01)

        self.click_if_exists("accept-all")

        # Get publisher
        try:
            publisher_locator = (By.XPATH, '//a[contains(@data-ylk, "sec:logo-provider")]')

            publisher_logo = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located(publisher_locator))
            print("Found")
        except TimeoutException as e:
            print("Publisher logo didn't appear within 10 seconds")
            # Return and do the db update outside
            raise e

        publisher_link = publisher_logo.get_attribute("href")
        print(publisher_link)

        # Get date
        times = self.driver.find_elements(By.TAG_NAME, "time")
        if len(times) != 1:
            print("Found an unexpected number of times")
        date = times[0].get_attribute("datetime")
        print(date)

        # Get full title
        # NOTE: If having trouble with this try using data-test-locator=headline instead
        #headline = self.driver.find_element(By.CLASS_NAME, "caas-lead-header-undefined").text
        headline = self.driver.find_element(By.CLASS_NAME, "cover-title").text
        print(headline)

        # Get related assets
        if "://finance." not in sanitized_url: # Localized Pages
            asset_buttons = self.driver.find_elements(By.CLASS_NAME, "caas-xray-pill-type-ticker")
            related_assets = [a.get_attribute("data-entity-id") for a in asset_buttons]
        else:
            #asset_buttons = self.driver.find_elements(By.TAG_NAME, "fin-ticker")
            #related_assets = [a.get_attribute("symbol") for a in asset_buttons]
            asset_buttons = self.driver.find_elements(By.CLASS_NAME, "ticker")
            related_assets = [a.get_attribute("title") for a in asset_buttons]

        # Get full text
        self.click_if_exists("collapse-button")
        self.click_if_exists("readmore-button")

        #news_body = self.driver.find_element(By.CLASS_NAME, "caas-body")
        news_body = self.driver.find_element(By.CLASS_NAME, "body")
        # TODO: Remove links to other articles
        paragraphs = news_body.find_elements(By.TAG_NAME, "p")
        full_text = "\n".join([p.text for p in paragraphs])
        print(full_text)

        article = Article(
            url=url,
            publisher_link=publisher_link,
            date=date,
            title=headline,
            assets=related_assets,
            full_text=full_text
        )

        return article
    
    def close(self):
        self.driver.quit()


def worker_task(worker: ScrapeWorker, url: str):
    try:
        return worker.scrape(url)
    except Exception as e:
        raise e
