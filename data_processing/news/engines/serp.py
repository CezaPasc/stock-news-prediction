import requests
import json
from urllib.parse import urlencode

import config

class SerpApi:

    def __init__(self) -> None:
        self.search_endpoint = "https://www.serpapi.com/search.json?"
        self.query = ""
        self.url = ""
        self.end_of_search = False

    def perform_search(self):
        assert self.url != "", "Please set a search query first using `set_search(query)`"

        if self.end_of_search:
            return []

        # Send the request
        response = requests.get(self.url, params={"api_key": config.serp_api_key})

        if response.status_code != 200:
            print("it happened")

        # Parse the JSON response
        data = response.json() 
        if "organic_results" not in data:
            return []

        results = data["organic_results"]

        if "next" in data["serpapi_pagination"]:
            self.url = data["serpapi_pagination"]["next"]
        else:
            self.end_of_search = True

        return results

    def set_search(self, query):
        self.end_of_search = False
        params = {
            "engine": "google",
            "q": f'site:finance.yahoo.com/news/ {query}',
            "num": 100,
            "filter": 0, # do not omit similar results
            "start": 0,
        }

        self.url = self.search_endpoint + urlencode(params)

    def next_page(self):
        return self.perform_search()
