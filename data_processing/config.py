import os
from dotenv import load_dotenv

"""
# Check if .env file exists in the current working directory
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(".env file loaded and environment variables set.")
else:
    print("No .env file found in the current directory.")
"""

data_dir = os.environ.get("DATA")
asset_dict = os.environ.get("ASSETS")
asset_naming = os.environ.get("ASSETS_NAMING")
risk_free_rates_path = os.environ.get("RISK_FREE_RATES")

# Price Cache
price_cache = os.environ.get("PRICE_CACHE")
only_cache = True if os.environ.get("PRICE_ONLY_CACHE", "False") == "True" else False

serp_api_key = os.environ.get("SERP_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")
google_se_id = os.environ.get("GOOGLE_ENGINE_ID")

reddit_client_id = os.environ.get("REDDIT_CLIENT_ID")
reddit_client_secret = os.environ.get("REDDIT_CLIENT_SECRET")

users_list = os.environ.get("TWITTER_USERS", "twitter_users.txt")

use_ner =  True if os.environ.get("USE_NER", "False") == "True" else False

huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")