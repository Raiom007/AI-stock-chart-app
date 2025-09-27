from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
import pandas as pd

from .data_handler import StockDataHandler
from .ai_controller import LocalLLMController

from utils.technical_indicators import TechnicalAnalyzer
from utils.pattern_recognition import ChartPatternRecognizer


app = FastAPI(title="AI Stock Chart API", version="1.0.1")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Limit this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_handler = StockDataHandler()
ai_controller = LocalLLMController()
pattern_recognizer = ChartPatternRecognizer()

class StockRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    timeframe: str = "1y"

class AnnotationRequest(BaseModel):
    symbol: str
    annotation_type: str
    coordinates: Dict
    user_query: str
    chart_data: Dict

    @field_validator('coordinates')
    def coordinates_must_have_start_end(cls, v):
        if not isinstance(v, dict) or 'start' not in v or 'end' not in v:
            raise ValueError('Coordinates must include start and end keys')
        return v

class IndicatorRequest(BaseModel):
    indicator_name: str = Field(..., min_length=1)
    value: float

@app.get("/")
async def root():
    return {"message": "AI Stock Chart API is running"}

@app.post("/api/stock/data")
async def get_stock_data(request: StockRequest):
    try:
        data = data_handler.get_stock_data_yfinance(request.symbol, request.timeframe)
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")
            
        analyzer = TechnicalAnalyzer(data)
        indicators = analyzer.calculate_all_indicators()
        patterns = analyzer.detect_patterns()
        levels = pattern_recognizer.detect_support_resistance(data)
        trend_lines = pattern_recognizer.detect_trend_lines(data)

        response_data = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "data": data.to_dict('records'),
            "indicators": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in indicators.items()},
            "patterns": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in patterns.items()},
            "support_resistance": levels,
            "trend_lines": trend_lines,
            "current_price": float(data['Close'].iloc[-1]),
            "volume": int(data['Volume'].iloc[-1])
        }
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/analyze_annotation")
async def analyze_annotation(request: AnnotationRequest):
    try:
        chart_data = request.chart_data or {}
        # Safely get required fields from chart_data
        current_price = chart_data.get("current_price")
        indicators = chart_data.get("indicators", {})
        volume = chart_data.get("volume")
        timeframe = chart_data.get("timeframe", "1d")
        support_levels = chart_data.get("support_levels", [])
        resistance_levels = chart_data.get("resistance_levels", [])
        recent_highs = chart_data.get("recent_highs", [])
        recent_lows = chart_data.get("recent_lows", [])
        raw_data_list = chart_data.get("raw_data", [])

        stock_context = {
            "symbol": request.symbol,
            "current_price": current_price,
            "indicators": indicators,
            "volume": volume,
            "timeframe": timeframe,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "recent_highs": recent_highs,
            "recent_lows": recent_lows
        }

        annotation_context = {
            "type": request.annotation_type,
            "coordinates": request.coordinates,
            "query": request.user_query
        }
        
        validation = None
        # Validate trendline coordinates safely
        if request.annotation_type == "trendline" and isinstance(raw_data_list, list) and len(raw_data_list) > 0:
            df = pd.DataFrame(raw_data_list)
            if not df.empty:
                coords = request.coordinates
                start_idx = coords.get('start', {}).get('x')
                end_idx = coords.get('end', {}).get('x')
                max_index = len(df) - 1
                if (
                    isinstance(start_idx, int) and 0 <= start_idx <= max_index and
                    isinstance(end_idx, int) and 0 <= end_idx <= max_index
                ):
                    validation = pattern_recognizer.validate_user_trendline(request.coordinates, df)

        ai_response = ai_controller.analyze_chart_annotation(annotation_context, stock_context)

        if validation:
            ai_response["validation"] = validation

        return ai_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/explain_indicator")
async def explain_indicator(request: IndicatorRequest):
    try:
        explanation = ai_controller.get_indicator_explanation(request.indicator_name, request.value)
        return {"indicator": request.indicator_name, "value": request.value, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    try:
        ollama_response = ai_controller._query_ollama("test")
        ollama_status = "up" if ollama_response and not ollama_response.startswith("Error") else "down"

        alpha_key_present = hasattr(data_handler, 'alpha_key') and data_handler.alpha_key is not None

        return {
            "status": "healthy",
            "ollama_llm": ollama_status,
            "data_sources": {
                "yfinance": "up",
                "alpha_vantage": "up" if alpha_key_present else "no_key"
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.websocket("/ws/real_time/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    await websocket.accept()
    try:
        import asyncio
        import pandas as pd
        while True:
            current_price = data_handler.get_real_time_price(symbol)
            if current_price is not None:
                await websocket.send_json({
                    "symbol": symbol,
                    "price": current_price,
                    "timestamp": pd.Timestamp.now().isoformat()
                })
            await asyncio.sleep(5)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

    