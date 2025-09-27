import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from typing import Dict, List


class ChartPatternRecognizer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def detect_support_resistance(self, data: pd.DataFrame, window=10) -> Dict:
        """Detect support and resistance levels"""
        highs = data['High'].rolling(window=window).max()
        lows = data['Low'].rolling(window=window).min()
        
        # Find significant highs and lows
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(data) - window):
            if data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append({
                    'price': data['High'].iloc[i],
                    'date': data.index[i],
                    'strength': self._calculate_level_strength(data, data['High'].iloc[i], 'resistance')
                })
                
            if data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append({
                    'price': data['Low'].iloc[i],
                    'date': data.index[i],
                    'strength': self._calculate_level_strength(data, data['Low'].iloc[i], 'support')
                })
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
        
    def detect_trend_lines(self, data: pd.DataFrame) -> Dict:
        """Detect trend lines using linear regression"""
        prices = data['Close'].values
        dates = np.arange(len(prices))
        
        # Find local highs and lows
        highs_idx = []
        lows_idx = []
        
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                if prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                    highs_idx.append(i)
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                if prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                    lows_idx.append(i)
        
        trend_lines = []
        
        # Connect significant highs and lows
        if len(highs_idx) >= 2:
            for i in range(len(highs_idx) - 1):
                slope = (prices[highs_idx[i+1]] - prices[highs_idx[i]]) / (highs_idx[i+1] - highs_idx[i])
                trend_lines.append({
                    'type': 'resistance_trendline',
                    'start': {'x': highs_idx[i], 'y': prices[highs_idx[i]]},
                    'end': {'x': highs_idx[i+1], 'y': prices[highs_idx[i+1]]},
                    'slope': slope
                })
        
        if len(lows_idx) >= 2:
            for i in range(len(lows_idx) - 1):
                slope = (prices[lows_idx[i+1]] - prices[lows_idx[i]]) / (lows_idx[i+1] - lows_idx[i])
                trend_lines.append({
                    'type': 'support_trendline',
                    'start': {'x': lows_idx[i], 'y': prices[lows_idx[i]]},
                    'end': {'x': lows_idx[i+1], 'y': prices[lows_idx[i+1]]},
                    'slope': slope
                })
        
        return {'trend_lines': trend_lines}
        
    def _calculate_level_strength(self, data: pd.DataFrame, level: float, level_type: str) -> int:
        """Calculate the strength of support/resistance level"""
        touches = 0
        tolerance = level * 0.01  # 1% tolerance
        
        if level_type == 'support':
            touches = len(data[abs(data['Low'] - level) <= tolerance])
        else:
            touches = len(data[abs(data['High'] - level) <= tolerance])
            
        return min(touches, 5)  # Max strength of 5

    def validate_user_trendline(self, coordinates: Dict, data: pd.DataFrame) -> Dict:
        """Validate if user-drawn trendline is technically sound"""
        start_x, start_y = coordinates['start']['x'], coordinates['start']['y']
        end_x, end_y = coordinates['end']['x'], coordinates['end']['y']
        
        # Check if trendline connects significant highs/lows
        tolerance = data['Close'].std() * 0.5
        
        validation_score = 0
        validation_reasons = []
        
        # Check if points are near actual highs/lows
        for idx in [start_x, end_x]:
            if idx < len(data):
                actual_high = data['High'].iloc[idx]
                actual_low = data['Low'].iloc[idx]
                
                if abs(start_y - actual_high) <= tolerance or abs(start_y - actual_low) <= tolerance:
                    validation_score += 1
                    validation_reasons.append(f"Point at index {idx} aligns with significant high/low")
        
        # Calculate slope and check for reasonable trend
        slope = (end_y - start_y) / (end_x - start_x) if end_x != start_x else 0
        
        if abs(slope) > 0.1:  # Significant trend
            validation_score += 1
            validation_reasons.append("Trendline shows significant directional bias")
            
        return {
            'is_valid': validation_score >= 2,
            'score': validation_score,
            'reasons': validation_reasons,
            'slope': slope,
            'type': 'uptrend' if slope > 0 else 'downtrend' if slope < 0 else 'sideways'
        }
