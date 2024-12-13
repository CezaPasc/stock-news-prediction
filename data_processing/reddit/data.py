import time
import os

import pandas as pd
import praw
from tqdm import tqdm

import config
import utils

def add_posts(posts, new_posts):
    len_posts = len(posts)
    # Retrieve posts
    new_posts = list(new_posts)

    # Add new posts
    posts += new_posts
    posts = list(set(posts))

    # Check how many new posts we got
    diff = len(posts) - len_posts
    print("Got %d new posts out of %d retrieved posts" % (diff, len(new_posts)))

    return posts

def add_category(subreddit, category, posts):
    print("Requesting posts from Subreddit %s, Category %s" % (subreddit.display_name, category))
    m = getattr(subreddit, category)
    c_posts = m()
    return add_posts(posts, c_posts)

def scrape_subreddit(subreddit, flairs, assets, only_search=False):
    posts = []

    flair_search_strings = [f'flair:"{f}"' for f in flairs]
    search_strings = flair_search_strings + assets

    # Get posts for requested flairs
    for search in tqdm(search_strings):
        for sort in ["relevance", "hot", "top", "new", "comments"]:
            # TODO: Maybe consider also varying the time filter
            print("Requesting posts from Subreddit %s, Flair %s, Sort: %s" % (subreddit.display_name, search, sort))
            flair_posts = subreddit.search(search, sort=sort, limit=None)       
            posts = add_posts(posts, flair_posts)

    if only_search:
        return posts

    for category in ["controversial", "hot", "new", "top", "rising"]:
        posts = add_category(subreddit, category, posts)
        
    return posts


def crawl():
    path = os.path.join(config.data_dir, "reddit/new_data.csv")
    # Create a Reddit instance
    reddit = praw.Reddit(client_id=config.reddit_client_id,
                        client_secret=config.reddit_client_secret,
                        user_agent='pycharm:com.example.safari:v1.2.3')


    assets = list(set(utils.get_asset_dict().values()))
    
    # Access a specific subreddit
    topics = {
        "WallStreetBets": {
            "flairs": ["News"],
            "onlySearch": True,
        },
        "stocks": {
            "flairs": ["Company News"],
            "assets": assets
        },
        "StockMarket": {
            "flairs": ["News"],
            "onlySearch": True
        }
        #"options": []
    }


    already_scraped_data = pd.read_csv(path)
    already_scraped_data = already_scraped_data.set_index("id", drop=False)

    for subreddit_name, requested in topics.items():
        subreddit = reddit.subreddit(subreddit_name)
        posts = scrape_subreddit(subreddit, requested["flairs"], requested.get("assets", []), requested.get("onlySearch", False))
        for post in posts:
            if post.id in already_scraped_data.index:
                continue

            created = post.created_utc
            age_when_scraped = time.time() - created
            entry = {
                    "id": post.id,
                    "title": post.title,
                    "shortlink": post.shortlink,
                    "url": post.url,
                    "author": str(post.author),
                    "created": created*1e9,
                    "created_ts": created,
                    "subreddit": subreddit_name,
                    "flair": post.link_flair_text,
                    "age_when_scraped": age_when_scraped,
                    "upvotes": post.ups,
                    "downvotes": post.downs,
                    "num_comments": post.num_comments,
                    "num_crossposts": post.num_crossposts,
                    "body": post.selftext,
                }
            already_scraped_data.loc[post.id] = entry


    already_scraped_data['created'] = pd.to_datetime(already_scraped_data['created'], format="mixed", utc=True)
    already_scraped_data.to_csv(path, index=False)
    print(already_scraped_data)
