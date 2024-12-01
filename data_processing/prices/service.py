from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import time

import config
from prices.alphavantage import AlphaVantageClient

class PriceService():

    def __init__(self, hours, excess_return=False, benchmark_index="QQQ", risk_free_rate_bond="^IRX"):
        self.period = hours
        self.av = AlphaVantageClient(cache_path=config.price_cache, only_cache=config.only_cache)
        self.return_method = self.get_excess_return if excess_return else self.get_asset_return
        # The order of this list can be adjusted to the preferred method for receiving the risk-free rate
        self.risk_free_methods = [self.retrieve_risk_free_csv, self.retrieve_risk_free_yfinance]


        self.benchmark_index = benchmark_index
        self.risk_free_rate_bond = risk_free_rate_bond
        
        self.risk_free_rates = None
        self.betas = {}
        self.last_yfinance_request = 0

    def get_price_development(self, asset, start_date, end_date):
        start_price = self.av.get_price(asset, start_date, alternative="last_known")
        end_price = self.av.get_price(asset, end_date, alternative="next_known")
        
        difference = end_price - start_price
        ror = difference / start_price   

        if abs(ror) > 0.5:
            print(f"{asset} has a big price difference between {start_date} (${start_price}) and {end_date} (${end_price}).\n Check if this is maybe due to a split.")
        
        return ror
    
    def retrieve_risk_free_yfinance(self, date):
        retries_left = 10
        bond_data = yf.Ticker(self.risk_free_rate_bond)
        rates = bond_data.history(start=date-timedelta(days=7), end=date)
        while rates.empty and retries_left > 0:
            date = date - timedelta(days=1)
            retries_left -= 1
            time.sleep(2)
            rates = bond_data.history(start=date-timedelta(days=30), end=date)

        if rates.empty:
            raise ValueError("Could not retrieve risk free rate")
    
        rate = rates["Close"].iloc[-1] / 100 
        return rate

    def retrieve_risk_free_csv(self, date):
        if self.risk_free_rates is None:
            # Load file
            self.risk_free_rates = pd.read_csv(config.risk_free_rates_path)
            self.risk_free_rates['Date'] = pd.to_datetime(self.risk_free_rates['Date'])
        
        date = date.tz_localize(None)

        valid_date_range = date - pd.DateOffset(months=1)
        matching_rates_df = self.risk_free_rates[(self.risk_free_rates['Date'] <= date) & 
                                                 (self.risk_free_rates['Date'] >= valid_date_range)]

        if  matching_rates_df.empty:
            raise ValueError(f"No risk-free rate saved for {date}")
        
        return matching_rates_df["Rate"].iloc[-1] / 100
        

    def get_risk_free_rate(self, date):
        annual_rate = None
        # Try to receive risk-free rates by trying its methods in preferred order
        for retrieve_risk_free in self.risk_free_methods:
            try:
                annual_rate = retrieve_risk_free(date)
                if annual_rate is not None:
                    break
            except Exception as e:  # Specify exceptions
                print(f"Failed to retrieve risk free rate using method {retrieve_risk_free}: {e}")
                continue
        else:
            raise ValueError("Could not retrieve risk free rate")
 
        # Break down risk-free rate to the specified period
        period_rate = (1 + annual_rate) ** (self.period / (365*24)) - 1
        
        return period_rate
    
    def get_beta(self, asset):
        if asset in self.betas:
            return self.betas[asset]

        # Cooldown Time
        now = time.time()
        difference = now - self.last_yfinance_request
        if difference < 3:
            to_wait = 3 - difference
            print("Waiting %s before next yfinance request" % to_wait)
            time.sleep(to_wait)
        
        beta = yf.Ticker(asset).info.get("beta", 1)
        self.last_yfinance_request = time.time()
        
        self.betas[asset] = beta
        
        return beta

    def get_excess_return(self, asset, start_date, end_date):
        try:
            asset_return = self.get_price_development(asset, start_date, end_date)
        except Exception as e:
            print(e)
            return pd.NA
            
        try:
            index_return = self.get_price_development(self.benchmark_index, start_date, end_date)
            risk_free_rate = self.get_risk_free_rate(end_date)
            beta = self.get_beta(asset)
        except Exception as e:
            print(e)
            return {"target": asset_return} 
        
        excess_return = asset_return - (risk_free_rate + beta * (index_return - risk_free_rate))

        return {
            "target": asset_return,
            "target_excess": excess_return
        }

    def get_asset_return(self, asset, start_date, end_date):
        try:
            development = self.get_price_development(asset, start_date, end_date)
        except Exception as e:
            print(e)
            development = pd.NA

        return {
            "target": development
        }

    def retrieve_price_development(self, post):
        start_date = post["created"]
        end_date = start_date + timedelta(hours=self.period)
        development = self.return_method(post["ticker"], start_date, end_date)
        return development


    def retrieve_price_development_for_periods(self, post, periods=[1,12,24]):
        development_periods = {}
        initial_period = self.period
        
        start_date = post["created"]
        for p in periods:
            self.period = p
            end_date= start_date + timedelta(hours=p)
            development = self.return_method(post["ticker"], start_date, end_date)
            if development is pd.NA:
                continue
            
            # Add period to key names
            development = {f"{k}_{p}h": v for k, v in development.items()}
            development_periods.update(development)
        
        self.period = initial_period
        return development_periods
