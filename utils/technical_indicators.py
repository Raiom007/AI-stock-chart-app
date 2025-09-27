import talib
import pandas as pd
import numpy as np

class TechnicalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.high = data['High'].values
        self.low = data['Low'].values
        self.close = data['Close'].values
        self.volume = data['Volume'].values
        
    def calculate_all_indicators(self):
        """Calculate common technical indicators"""
        indicators = {}
        
        # Moving Averages
        indicators['SMA_20'] = talib.SMA(self.close, timeperiod=20)
        indicators['EMA_20'] = talib.EMA(self.close, timeperiod=20)
        indicators['SMA_50'] = talib.SMA(self.close, timeperiod=50)
        
        # Momentum Indicators
        indicators['RSI'] = talib.RSI(self.close, timeperiod=14)
        indicators['MACD'], indicators['MACD_signal'], indicators['MACD_hist'] = talib.MACD(self.close)
        
        # Volatility
        indicators['BBANDS_upper'], indicators['BBANDS_middle'], indicators['BBANDS_lower'] = talib.BBANDS(self.close)
        indicators['ATR'] = talib.ATR(self.high, self.low, self.close, timeperiod=14)
        
        # Volume Indicators
        indicators['OBV'] = talib.OBV(self.close, self.volume)
        
        # Custom calculations
        indicators['volatility'] = self.calculate_volatility()
        indicators['daily_returns'] = self.calculate_daily_returns()
        indicators['sharpe_ratio'] = self.calculate_sharpe_ratio()
        
        return indicators
        
    def calculate_volatility(self):
        """Calculate price volatility"""
        returns = np.log(self.close[1:] / self.close[:-1])
        return np.std(returns) * np.sqrt(252)  # Annualized
        
    def calculate_daily_returns(self):
        """Calculate daily percentage returns"""
        return (self.close[1:] - self.close[:-1]) / self.close[:-1] * 100
        
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        returns = self.calculate_daily_returns()
        excess_returns = np.mean(returns) - risk_free_rate/252
        return excess_returns / np.std(returns) if np.std(returns) != 0 else 0
        
    def detect_patterns(self):
        """Detect candlestick patterns using TA-Lib"""
        patterns = {}
        
        # Bullish patterns
        patterns['hammer'] = talib.CDLHAMMER(self.data['Open'], self.high, self.low, self.close)
        patterns['doji'] = talib.CDLDOJI(self.data['Open'], self.high, self.low, self.close)
        patterns['engulfing_bull'] = talib.CDLENGULFING(self.data['Open'], self.high, self.low, self.close)
        
        # Bearish patterns
        patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(self.data['Open'], self.high, self.low, self.close)
        patterns['dark_cloud'] = talib.CDLDARKCLOUDCOVER(self.data['Open'], self.high, self.low, self.close)
        
        return patterns
