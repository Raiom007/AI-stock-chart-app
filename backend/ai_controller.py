import requests
import json
from typing import Dict, List

class LocalLLMController:
    def __init__(self, model_name="llama3.1:8b"):
        self.base_url = "http://localhost:11434"
        self.model_name = model_name
        
    def analyze_chart_annotation(self, annotation_data: Dict, stock_data: Dict) -> Dict:
        """Analyze user chart annotations with AI"""
        
        # Prepare context for LLM
        prompt = self._build_analysis_prompt(annotation_data, stock_data)
        
        # Query local LLM
        response = self._query_ollama(prompt)
        
        # Parse response for both text and visual instructions
        return self._parse_ai_response(response)
        
    def _build_analysis_prompt(self, annotation_data: Dict, stock_data: Dict) -> str:
        """Build comprehensive prompt for financial analysis"""
        
        current_price = stock_data.get('current_price', 'N/A')
        rsi = stock_data.get('indicators', {}).get('RSI', 'N/A')
        volume = stock_data.get('volume', 'N/A')
        
        prompt = f"""
        You are a financial analyst AI. Analyze the following stock chart annotation:
        
        STOCK DATA:
        - Symbol: {stock_data.get('symbol', 'Unknown')}
        - Current Price: ${current_price}
        - RSI: {rsi}
        - Volume: {volume}
        - Timeframe: {stock_data.get('timeframe', '1d')}
        
        USER ANNOTATION:
        - Type: {annotation_data.get('type', 'line')}  # line, trendline, rectangle, etc.
        - Coordinates: {annotation_data.get('coordinates', {})}
        - User Query: "{annotation_data.get('query', '')}"
        
        CHART CONTEXT:
        - Recent Highs: {stock_data.get('recent_highs', [])}
        - Recent Lows: {stock_data.get('recent_lows', [])}
        - Support Levels: {stock_data.get('support_levels', [])}
        - Resistance Levels: {stock_data.get('resistance_levels', [])}
        
        Please provide:
        1. Technical analysis of the annotation
        2. Whether it represents valid support/resistance
        3. Trading implications
        4. Suggested chart overlays (JSON format for frontend)
        
        Response format:
        {{
            "analysis": "Your technical analysis here",
            "validity": "high/medium/low",
            "trading_signal": "bullish/bearish/neutral", 
            "visual_updates": {{
                "add_indicators": ["SMA_20", "RSI"],
                "highlight_zones": [{{
                    "type": "support",
                    "price_level": 150.25,
                    "color": "green",
                    "opacity": 0.3
                }}],
                "add_annotations": []
            }}
        }}
        """
        
        return prompt
        
    def _query_ollama(self, prompt: str) -> str:
        """Query local Ollama LLM"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()['response']
            else:
                return "Error: Unable to get AI response"
        except Exception as e:
            return f"Error connecting to local LLM: {str(e)}"
            
    def _parse_ai_response(self, response: str) -> Dict:
        """Parse AI response into structured format"""
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_part = response[start:end]
                return json.loads(json_part)
            else:
                # Fallback to text-only response
                return {
                    "analysis": response,
                    "validity": "medium",
                    "trading_signal": "neutral",
                    "visual_updates": {}
                }
        except:
            return {
                "analysis": response,
                "validity": "low", 
                "trading_signal": "neutral",
                "visual_updates": {}
            }
            
    def get_indicator_explanation(self, indicator_name: str, value: float) -> str:
        """Get AI explanation for technical indicators"""
        prompt = f"""
        Explain the technical indicator {indicator_name} with current value {value}.
        Provide:
        1. What this indicator measures
        2. How to interpret the current value
        3. Trading implications
        4. What traders should watch for next
        
        Keep the explanation concise but informative for both beginners and experienced traders.
        """
        
        return self._query_ollama(prompt)
