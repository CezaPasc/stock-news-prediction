import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime

import config
import utils
from prices.service import PriceService

def reddit_preprocessing(df):
    # Combine body if necessary
    df['text'] = df['title'] + " " + df['body']
    df["created"] = pd.to_datetime(df["created"])
    df = df.dropna(subset=["text"])
    return df

def twitter_preprocessing(df, extract_tickers=False):
    df["created"] = pd.to_datetime(df["created_at"])
    df["ID"] = df["id_str"]
    
    if extract_tickers:
        from ast import literal_eval
        df["entitiesJSON"] = df["entities"].apply(literal_eval)
        df["tickers"] = df["entitiesJSON"].apply(lambda x: [i["text"] for i in x.get("symbols")] if isinstance(x, dict) else None)
        return df[["text", "created", "tickers"]]

    return df[["ID", "text", "created"]]
    

def social_media(df, platform="reddit", verbose_saving=False):
    if platform == "reddit":
        df = reddit_preprocessing(df)
    elif platform == "twitter":
        df = twitter_preprocessing(df)
    else:
        print(f"Platform {platform} is not known.")
        exit()
    
    # Extract tickers
    print("Extracting ticker names...")
    df["tickers"] = df['text'].progress_apply(utils.extract_tickers_selected)
    # Unify ticker names
    df["tickers"] = df["tickers"].apply(lambda x: [utils.asset_naming.get(val, val) for val in x])
    df["tickers"] = df["tickers"].apply(lambda x: list(set(x)))
    if verbose_saving:
        target_path = path.replace(".csv", "_with_tickers.csv")
        df.to_csv(target_path, index=False)

    return df

def news(df):
    df.rename(columns={"Date": "created"}, inplace=True)
    df["created"] = pd.to_datetime(df["created"])
    df["tickers"] = df["tickers"].apply(utils.parse_yahoo_tickers)
    df["tickers"] = df["tickers"].apply(lambda x: [utils.asset_naming.get(val, val) for val in x])
    df["tickers"] = df["tickers"].apply(lambda x: list(set(x)))

    return df


# Setup
kind = "news"
platform = "yahoo"
path = "yahoo/news_final.csv"
path = os.path.join(config.data_dir, path)

price_development_period = 1 # in hours
limit_stocks = True
single_asset = True # Set to true for creating datasets for single-asset training 


if __name__ == "__main__":
    tqdm.pandas()
    priceService = PriceService(price_development_period, excess_return=True)

    # Read a dataset
    df = pd.read_csv(path)

    # Preprocessing
    if kind == "socialmedia":
        df = social_media(df, platform, verbose_saving=True)
    elif kind == "news":
        df = news(df)


    # Create a pair for each post-asset combination
    pairs = utils.extract_news_asset_pairs(df, single_asset=single_asset)
    
    # Limit stocks (only necessary for news, since tickers of social media were extracted using stock selection)
    if limit_stocks and kind == "news":
        print("Deleting all news-asset pairs not belonging to selection of stocks...")
        len_before = len(pairs)
        stock_selection = set(utils.asset_dict.values())
        pairs = pairs[pairs["ticker"].isin(stock_selection)]
        deleted_obs = len_before - len(pairs)
        print("Deleted %d obs. %d -> %d" % (deleted_obs, len_before, len(pairs)))

    # Shuffle in order to make new requests when fetching prices
    pairs = pairs.sample(frac=1)
    
    # Calculate target data
    tqdm.pandas(desc="Generating target data")
    target = pairs.progress_apply(
        lambda row: priceService.retrieve_price_development(row),
        #lambda row: priceService.retrieve_price_development_for_periods(row), # calculating target for multipe periods
        axis=1,
        result_type="expand"
    )
    pairs = pairs.join(target)
    
    # Save file
    target_path = path.replace(".csv", "_target_%dh_%s.csv" % (price_development_period, datetime.now().strftime("%Y-%m-%d")))
    pairs.to_csv(target_path, index=False)
    print("Saved data to %s" % target_path)
