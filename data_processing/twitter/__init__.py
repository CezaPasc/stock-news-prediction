import pandas as pd
from datetime import datetime

from twitter import timeline
import config

def crawl():
    # Read users file
    with open(config.users_list, 'r') as file:
        users = [l.strip().split("/")[-1] for l in file]
        
    tweets = []
    for user in users:
        tweets += timeline.get_tweets_of_user(user)

    df = pd.DataFrame(tweets) # Some processing to the nested columns could be helpful here
    
    output_name = "tweets_export_%s.csv" % datetime.now().strftime("%Y-%m-%d")
    df.to_csv(output_name)
