import requests
import json

import config

class GoogleCSE:

    def __init__(self) -> None:
        self.url_template = 'https://www.googleapis.com/customsearch/v1?q={searchTerms}&num={count?}&start={startIndex?}&lr={language?}&safe={safe?}&cx={cx?}&sort={sort?}&filter={filter?}&gl={gl?}&cr={cr?}&googlehost={googleHost?}&c2coff={disableCnTwTranslation?}&hq={hq?}&hl={hl?}&siteSearch={siteSearch?}&siteSearchFilter={siteSearchFilter?}&exactTerms={exactTerms?}&excludeTerms={excludeTerms?}&linkSite={linkSite?}&orTerms={orTerms?}&dateRestrict={dateRestrict?}&lowRange={lowRange?}&highRange={highRange?}&searchType={searchType}&fileType={fileType?}&rights={rights?}&imgSize={imgSize?}&imgType={imgType?}&imgColorType={imgColorType?}&imgDominantColor={imgDominantColor?}&alt=json'
        self.next_index = 1
        self.query = ""
        self.end_of_search = False

    def perform_search(self):
        if self.end_of_search:
            return []
        
        url = f"https://www.googleapis.com/customsearch/v1?q={self.query}&key={config.google_api_key}&cx={config.google_se_id}&num=10&start={self.next_index}"

        # Send the request
        response = requests.get(url)

        if response.status_code != 200:
            return []

        # Parse the JSON response
        data = json.loads(response.text)

        if "nextPage" in data["queries"]:
            next_page = data["queries"]["nextPage"]
            self.next_index = next_page[0]["startIndex"]
        else:
            self.end_of_search = True

        return data["items"]

    def set_search(self, query):
        self.end_of_search = False
        self.next_index = 1
        self.query = query

    def next_page(self):
        return self.perform_search()
