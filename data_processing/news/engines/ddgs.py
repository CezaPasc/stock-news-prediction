from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException
import time

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

class DuckDuckGo:

    def __init__(self, limit=5, proxy=None) -> None:
        self.limit = limit
        self.query = ""
        self.duck = DDGS(proxy=proxy)
        self.new_search = False
        self.last_request = 0
        self.cooldown = 30 

    def perform_search(self):
        assert self.query != "", "Please set a search query first using `set_search(query)`"
        now = time.time()
        difference = now - self.last_request
        if difference < self.cooldown:
            to_wait = self.cooldown - difference
            print("Waiting %s before next request" % to_wait)
            time.sleep(to_wait)

        # Send the request
        try:
            results = self.duck.text(self.query, max_results=self.limit)
        except RatelimitException as e:
            print(e)
            time.sleep(3)
            print("Initialsing new Duck")
            self.duck = DDGS() 
            print("Starting new try")
            return self.perform_search()
        except Exception as e:
            print("Some other exception has occurred")
            print(e)
            yes = True
            if yes:
                exit()
            
        self.new_search = False
        self.last_request = time.time()

        for result in results:
            result["link"] = result.pop("href") 

        return results

    def set_search(self, query):
        self.query = f'site:finance.yahoo.com/news/ {query}'
        self.new_search = True

    def next_page(self):
        if not self.new_search:
            return []
        
        return self.perform_search()


if __name__ == "__main__":
    duck = DuckDuckGo()
    
    while True:
        duck.set_search("some")
        results = duck.perform_search()

        duck.set_search("search")
        results = duck.perform_search()

        duck.set_search("terms")
        results = duck.perform_search()
