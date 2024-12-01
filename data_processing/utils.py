import config
import re
import json

import pandas as pd

pattern = r'\b[A-Z]{3,5}\b'
conservative_pattern = r'\$\b[A-Za-z]{3,5}\b'

# Testing
def get_bigrams_dict(dict) -> dict:
    bigrams = {}
    for key in dict.keys():
        bigrams[key] = get_bigrams(key)
    
    return bigrams

def get_bigrams(string):
    s = string.lower()
    return [s[i : i + 2] for i in range(len(s) - 1)]

def simon_similarity(pairs1, pairs2):
    common_pairs = set(pairs1) & set(pairs2)
    similarity = len(common_pairs) / max(len(pairs1), len(pairs2))
    return similarity
# TEsting


def get_asset_dict() -> dict:
    with open(config.asset_dict, 'r') as f:
        return json.load(f)
 
def get_asset_naming() -> dict:
    with open(config.asset_naming, 'r') as f:
        return json.load(f)

def get_embeddings(dict) -> dict:
    embeddings = {}
    for key in dict.keys():
        embeddings[key] = nlp(key)
    
    return embeddings


asset_dict = get_asset_dict()
asset_naming = get_asset_naming()
banned_words = ["CEO", "TLDR"]

if config.use_ner:
    import spacy
    nlp = spacy.load('en_core_web_md')
    asset_embeddings = get_embeddings(asset_dict)
    asset_bigrams = get_bigrams_dict(asset_dict)


def extract_tickers(text) -> list[str]:
    matches = re.findall(pattern, text)

    unique_tickers = set(matches)
    [unique_tickers.discard(w) for w in banned_words]
    return list(unique_tickers)

def extract_tickers_conservative(text) -> list[str]:
    matches = re.findall(conservative_pattern, text)
    tickers_without_dollar = [match[1:].upper() for match in matches]

    unique_tickers = set(tickers_without_dollar)
    return list(unique_tickers)





def extract_tickers_selected(text) -> list[str]:
    found_words = []

    if config.use_ner:
        entities = [e for e in nlp(text).ents if e.label_ == "ORG"]

    for search_word, ticker in asset_dict.items():
        # Search can be skipped, when the ticker of the search word is already found in the text
        if ticker in found_words:
            continue

        # The pattern ensures the word is not followed by an alphabetic character
        pattern = re.compile(r'\b' + re.escape(search_word) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            found_words.append(ticker)
            continue
        
        if not config.use_ner:
            continue

        for entity in entities:
            search_token = asset_embeddings[search_word]
            similarity_flair = search_token.similarity(entity)
            if similarity_flair > 0.999:
                print("Similarity between '%s' and '%s' is %f" % (search_word, entity.text, similarity_flair))

            search_bigram = asset_embeddings[search_word]
            entity_bigram = get_bigrams(entity.text)
            similarity_bigram = simon_similarity(search_bigram, entity_bigram)
            if similarity_bigram > 0:
                print("Similarity between '%s' and '%s' is %f" % (search_word, entity.text, similarity_bigram))


    return found_words

def extract_news_asset_pairs(dataframe, single_asset=False):
    if single_asset:
        dataframe = dataframe[dataframe["tickers"].apply(lambda x: len(x)==1)]
        
    pairs = dataframe.explode("tickers", ignore_index=True)
    pairs.rename(columns={"tickers": "ticker"}, inplace=True)

    # Exclude rows without assets
    pairs = pairs.dropna(subset=["ticker"])

    return pairs


def parse_yahoo_tickers(ticker_str):
    if pd.isna(ticker_str):
        return []
    else:
        tickers =  ticker_str.split(',') 
        tickers = [t.strip() for t in tickers]
        return tickers
