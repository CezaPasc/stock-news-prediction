import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time
import os
import warnings
import string
import random

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

# Date parser of pd.read_csv (version 2.2.1) is deprecated, but its alternative can not achieve the old functionality
warnings.simplefilter(action='ignore', category=FutureWarning)

TOLERANCE = timedelta(hours=1)
WEEKEND_TOLERANCE = timedelta(days=2, hours=8, minutes=30)
COOLDOWN = 5 # seconds
MAX_RECURSION_DEPTH = 3

naming_convention = {
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. volume": "volume"
}

class AlphaVantageClient():

    def __init__(self, cache_path, base_url="https://www.alphavantage.co", only_cache=False) -> None:
        self.api_key = "".join(random.choice(string.ascii_uppercase+string.digits) for _ in range(16))
        self.base_url = base_url
        self.cache_path = cache_path
        self.last_request = 0
        self.cached = {}
        self.only_cache = only_cache
        # The dates are provided in the timezone of ET: https://www.alphavantage.co/documentation/#intraday
        self.tz = "America/New_York"

    def custom_date_parser(self, x):
        try:
            # If the first attempt fails, try parsing with the format "YYYY-MM-DD-HH-MM-SS+00:00"
            return pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S%z', utc=True)
        except ValueError as e:
            dates = pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
            dates = dates.tz_localize(self.tz).tz_convert("UTC")
            return dates

    def is_cached(self, asset: str, date: datetime) -> bool:
        if asset not in self.cached:
            print("Try to load saved price data for %s" % asset)
            try:
                path = os.path.join(self.cache_path, "%s.csv" % asset)
                df = pd.read_csv(path, parse_dates=["date"], date_parser=self.custom_date_parser, index_col="date")
                df.sort_index(inplace=True)
                self.cached[asset] = df
            except FileNotFoundError:
                return False
            except Exception as e:
                print("Error while reading CSV")
                print(e)
        
        # Try to find given date in already retrieved asset prices
        asset_df = self.cached[asset]
        date_df = asset_df[date - WEEKEND_TOLERANCE: date + WEEKEND_TOLERANCE]
        if date_df.empty:
            return False
        
        is_in_month = (date_df.index.month == date.month).any()

        return not date_df.empty and is_in_month

    def update_cache(self, asset, prices):
        prices.rename(naming_convention, axis=1, inplace=True)
        prices.index = pd.to_datetime(prices.index).tz_localize(self.tz).tz_convert("UTC")
        prices.index.name = 'date'
        prices = prices.astype('float')
        
        # Add to cache
        if asset in self.cached:
            self.cached[asset] = pd.concat([self.cached[asset], prices])
        else:
            self.cached[asset] = prices
        
        self.cached[asset].sort_index(inplace=True)
        self.cached[asset] = self.cached[asset].loc[~self.cached[asset].index.duplicated(keep='first')]

        # Save to disk
        path = os.path.join(self.cache_path, "%s.csv" % asset)
        self.cached[asset].to_csv(path)


    def request(self, asset, month) -> dict:
        print("Requesting prices of %s at %s" % (asset, month))
        # Cooldown Time
        now = time.time()
        difference = now - self.last_request
        if difference < COOLDOWN:
            to_wait = COOLDOWN - difference
            print("Waiting %s before next request" % to_wait)
            time.sleep(to_wait)

        endpoint = 'query'
        url = '%s/%s' % (self.base_url, endpoint)
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": asset,
            "interval": "1min",
            "month": month,
            "outputsize": "full",
            "apikey": self.api_key
        }
        try:
            res = requests.get(url, params)
        except Exception as e:
            print("Error requesting prices")
            print(e)
            time.sleep(3)
            return self.request(asset, month)
           
        self.last_request = time.time()
        if res.status_code != 200:
            print("Some error happened while requesting intraday data from")
            print("Status Code:", res.status_code)
            print("Error message: ")
            print(res.text)
            return {}
        
        data = res.json()    
        # Success case
        if "Time Series (1min)" in data:
            prices = data["Time Series (1min)"]
            return prices
        
        # Notice about e.g. rate-limit
        if "Information" in data:
            print("AlphaVantage didn't return the expected response")
            print(data)
            time.sleep(3)
            return self.request(asset, month)
        
        # Error case when no data is available
        if "Error Message" in data:
            print("An error occured:")
            print(data["Error Message"])
            return {}
        
    def fetch_month(self, asset, date):
        # Request Intraday Data
        month = date.strftime("%Y-%m")
        prices = self.request(asset, month)
        # Process and cache response
        prices_df = pd.DataFrame(prices).transpose()
        self.update_cache(asset, prices_df)

    def get_price(self, asset, date: datetime, alternative=None, recursion_depth=0) -> float:
        if recursion_depth > MAX_RECURSION_DEPTH:
            raise RecursionError("Max recursion depth reached.")
        
        # Check if date is valid
        yesterday = datetime.now() - timedelta(days=1)
        if date.timestamp() > yesterday.timestamp():
            raise ValueError(f"The requested date is in the future.")
        
        # Fetch prices on demand if not cached
        if not self.is_cached(asset, date) and not self.only_cache:
            self.fetch_month(asset, date)          

        # Get the nearest price to the requested date 
        asset_df = self.cached[asset]
        date_df = asset_df[date - WEEKEND_TOLERANCE: date + WEEKEND_TOLERANCE]
        if date_df.empty:
            raise ValueError(f'There is no price available for {asset} on {date} within an acceptable range.')
        
        nearest_datetime = min(date_df.index, key=lambda x: abs(x - date))
        time_difference = abs(nearest_datetime - date)

        # Check if the time difference is bigger than a weekend. In this case probably data is missing or we have a weekend followed by holidays
        if time_difference > WEEKEND_TOLERANCE:
            raise ValueError(f"The nearest price ({nearest_datetime}) is too far from the target datetime ({date}).")

        if time_difference > TOLERANCE:
            if not alternative:
                raise ValueError(f"The nearest price ({nearest_datetime}) is too far from the target datetime ({date}).")

            if alternative == "last_known":
                if nearest_datetime > date:
                    # Take the latest known price at the date
                    nearest_loc = asset_df.index.get_loc(nearest_datetime)
                    if nearest_loc -1 < 0:
                        if self.only_cache:
                            raise LookupError("Requesting of prices disabled. Please turn off cache_only")

                        # Request previous month
                        previous_month = date + pd.DateOffset(months=-1)
                        self.fetch_month(asset, previous_month)
                        # Call method again, so that the closest price can be reevaluated again
                        return self.get_price(asset, date, alternative, recursion_depth+1)

                    nearest_datetime = asset_df.index[nearest_loc-1]

            if alternative == "next_known":
                if date > nearest_datetime:
                    nearest_loc = asset_df.index.get_loc(nearest_datetime)
                    if nearest_loc+1 >= len(asset_df):
                        if self.only_cache:
                            raise LookupError("Requesting of prices disabled. Please turn off cache_only")

                        # Request next month
                        next_month = date + pd.DateOffset(months=1)
                        self.fetch_month(asset, next_month)
                        # Call method again, so that the closest price can be reevaluated again
                        return self.get_price(asset, date, alternative, recursion_depth+1)
                    
                    nearest_datetime = asset_df.index[nearest_loc+1]
                # TODO: it would be an option to add one hour grace time, since this is the first minute of pre market trading (this is probalby only important for weekend news, but not midnight news)

        nearest_candle = asset_df.loc[nearest_datetime]
        return nearest_candle["close"]
    

if __name__ == "__main__":
    av = AlphaVantageClient(
        cache_path="../data/intraday"
    )
    
    def request_test_prices():
        # Test Cases
        p = av.get_price("AAPL", datetime(year=2021, month=4, day=28, hour=12, minute=20, tzinfo=ZoneInfo("America/New_York")), alternative="last_known")
        p = av.get_price("AMD", datetime(year=2022, month=4, day=15, hour=13, minute=20, tzinfo=ZoneInfo("America/New_York")), alternative="last_known")
        p = av.get_price("MSFT", datetime(year=2022, month=2, day=16, hour=14, minute=20, tzinfo=ZoneInfo("America/New_York")), alternative="last_known")
        p = av.get_price("AAPL", datetime(year=2023, month=6, day=20, hour=15, minute=20, tzinfo=ZoneInfo("America/New_York")), alternative="last_known")
        p = av.get_price("GOOGL", datetime(year=2023, month=7, day=24, hour=10, minute=20, tzinfo=ZoneInfo("America/New_York")), alternative="last_known")
        p = av.get_price("IBM", datetime(year=2022, month=2, day=22, hour=11, minute=20, tzinfo=ZoneInfo("America/New_York")), alternative="last_known")
        p = av.get_price("GOOGL", datetime(year=2023, month=7, day=12, hour=12, minute=20, tzinfo=ZoneInfo("America/New_York")), alternative="last_known")
        
        # Error Cases
        p = av.get_price("ABNB", datetime(year=2020, month=12, day=9, hour=23, minute=46, tzinfo=ZoneInfo("America/New_York")), alternative="last_known")
        p = av.get_price("QQQ", datetime(year=2020, month=12, day=9, hour=23, minute=46, tzinfo=ZoneInfo("America/New_York")), alternative="last_known")
    
    start_time = time.time()
    request_test_prices()
    execution_time = time.time() - start_time
    print("Execution time in seconds (with loading): " + str(execution_time))

    start_time = time.time()
    request_test_prices()
    execution_time = time.time() - start_time
    print("Execution time in seconds (already loaded): " + str(execution_time))
    
