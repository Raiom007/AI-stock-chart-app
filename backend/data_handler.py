import yfinance as yf
import requests
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv

load_dotenv()

class StockDataHandler:
    def __init__(self):
        self.alpha_key = os.getenv('60QSLAV4MVGYMDGP')
        self.ts = TimeSeries(key=self.alpha_key, output_format='pandas')
        
    def get_stock_data_yfinance(self, symbol, period="1y"):
        """Primary free data source - unlimited requests"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"YFinance error: {e}")
            return None
            
    def get_stock_data_alpha(self, symbol, interval='1day'):
        """Secondary data source - 500 requests/day"""
        try:
            data, meta_data = self.ts.get_daily(symbol=symbol, outputsize='full')
            return data
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
            return None
            
    def get_real_time_price(self, symbol):
        """Get current price using yfinance (free)"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return info['lastPrice']
        except:
            return None
            
    def get_technical_indicators_alpha(self, symbol, indicator='RSI'):
        """Get pre-computed technical indicators"""
        url = f'https://www.alphavantage.co/query'
        params = {
            'function': indicator,
            'symbol': symbol,
            'interval': 'daily',
            'time_period': 14,
            'series_type': 'close',
            'apikey': self.alpha_key
        }
        
        response = requests.get(url, params=params)
        return response.json()
