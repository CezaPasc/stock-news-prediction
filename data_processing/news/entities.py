
class Article:

    def __init__(self, url, publisher_link, date, title, assets, full_text) -> None:
        self.url = url
        self.publisher_link = publisher_link
        self.date = date
        self.title = title
        self.assets = assets
        self.full_text = full_text
