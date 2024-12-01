#!/usr/bin/python3
# Inspiration: https://github.com/zedeus/nitter/issues/983#issuecomment-1678942933
import requests
import re
import urllib
import json

def get_tweets_of_user(username: str) -> list[dict]:
    url  = f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{username}"

    with urllib.request.urlopen(url) as response:
        encoding = response.info().get_param('charset', 'utf8')
        html = response.read().decode(encoding)
        result = re.search('script id="__NEXT_DATA__" type="application\/json">([^>]*)<\/script>', html)[1]

        res = json.loads(result)
        tweets = res['props']['pageProps']['timeline']['entries']
        tweets = [t['content']['tweet'] for t in tweets if "tweet" in t['content']]
        return tweets