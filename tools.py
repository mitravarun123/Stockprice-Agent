import os
import json
import feedparser
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_anthropic import ChatAnthropic




@tool("Fetch Price Tool")
def fetch_price_tool(ticker: str) -> str:
    """Fetch real-time stock price, P/E ratio, market cap, 52-week range from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        hist  = stock.history(period="5d")

        if hist.empty:
            return f"ERROR: No data for {ticker}"

        current    = round(float(hist["Close"].iloc[-1]), 2)
        prev       = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else current
        change     = round(current - prev, 2)
        change_pct = round((change / prev) * 100, 2)

        data = {
            "ticker":     ticker,
            "company":    info.get("longName", ticker),
            "price":      current,
            "change":     change,
            "change_pct": change_pct,
            "market_cap": info.get("marketCap"),
            "pe_ratio":   info.get("trailingPE"),
            "52w_high":   info.get("fiftyTwoWeekHigh"),
            "52w_low":    info.get("fiftyTwoWeekLow"),
            "sector":     info.get("sector", "N/A"),
            "volume":     info.get("volume"),
            "beta":       info.get("beta"),
            "div_yield":  info.get("dividendYield"),
        }
        return json.dumps(data)
    except Exception as e:
        return f"ERROR:{str(e)}"


@tool("Fetch News Tool")
def fetch_news_tool(ticker: str) -> str:
    """Fetch latest stock news headlines from Yahoo Finance RSS and Finnhub."""
    headlines = []
    try:
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        for entry in feed.entries[:6]:
            headlines.append({
                "title":     entry.get("title", ""),
                "published": entry.get("published", ""),
                "source":    "Yahoo Finance"
            })
    except Exception as e:
        return f"ERROR:{str(e)}"

    finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    if finnhub_key and finnhub_key != "your_key":
        try:
            to_d   = datetime.now().strftime("%Y-%m-%d")
            from_d = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            resp   = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": ticker, "from": from_d, "to": to_d, "token": finnhub_key},
                timeout=10
            )
            if resp.status_code == 200:
                for item in resp.json()[:4]:
                    headlines.append({
                        "title":  item.get("headline", ""),
                        "source": "Finnhub"
                    })
        except Exception:
            pass

    return json.dumps(headlines)


@tool("Fetch Technical Tool")
def fetch_technical_tool(ticker: str) -> str:
    """Calculate RSI, MACD, SMA20, SMA50, Bollinger Bands and ATR for a stock."""
    try:
        stock  = yf.Ticker(ticker)
        hist   = stock.history(period="6mo")

        if hist.empty or len(hist) < 20:
            return "ERROR: Not enough data"

        closes = hist["Close"]
        highs  = hist["High"]
        lows   = hist["Low"]

        # SMA
        sma20 = round(float(closes.tail(20).mean()), 2)
        sma50 = round(float(closes.tail(50).mean()), 2) if len(closes) >= 50 else None

        # RSI
        delta = closes.diff()
        gain  = delta.clip(lower=0).tail(14).mean()
        loss  = (-delta.clip(upper=0)).tail(14).mean()
        rsi   = round(float(100 - (100 / (1 + gain / loss))), 1) if loss != 0 else 50.0

        # MACD
        ema12  = closes.ewm(span=12).mean()
        ema26  = closes.ewm(span=26).mean()
        macd   = round(float((ema12 - ema26).iloc[-1]), 4)
        signal = round(float((ema12 - ema26).ewm(span=9).mean().iloc[-1]), 4)

        # Bollinger Bands
        rolling_mean = closes.rolling(20).mean()
        rolling_std  = closes.rolling(20).std()
        bb_upper = round(float((rolling_mean + 2 * rolling_std).iloc[-1]), 2)
        bb_lower = round(float((rolling_mean - 2 * rolling_std).iloc[-1]), 2)

        # ATR
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift()).abs(),
            (lows  - closes.shift()).abs()
        ], axis=1).max(axis=1)
        atr = round(float(tr.tail(14).mean()), 2)

        # 1-month return
        one_month = round(float(((closes.iloc[-1] / closes.iloc[-21]) - 1) * 100), 2) if len(closes) >= 21 else None

        # Trend and signals
        trend = "UPTREND" if (sma50 and sma20 > sma50) else "DOWNTREND"
        rsi_label = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"

        signals = []
        if rsi < 30:      signals.append("RSI OVERSOLD - bullish signal")
        if rsi > 70:      signals.append("RSI OVERBOUGHT - bearish signal")
        if macd > signal: signals.append("MACD bullish crossover")
        else:             signals.append("MACD bearish crossover")
        if sma50 and sma20 > sma50: signals.append("Golden cross - bullish")
        else:                        signals.append("Death cross - bearish")

        data = {
            "rsi":           rsi,
            "rsi_label":     rsi_label,
            "macd":          macd,
            "macd_signal":   signal,
            "sma20":         sma20,
            "sma50":         sma50,
            "bb_upper":      bb_upper,
            "bb_lower":      bb_lower,
            "atr":           atr,
            "trend":         trend,
            "one_month_pct": one_month,
            "signals":       signals,
        }
        return json.dumps(data)
    except Exception as e:
        return f"ERROR:{str(e)}"


@tool("Fetch Analyst Ratings Tool")
def fetch_analyst_ratings_tool(ticker: str) -> str:
    """Fetch Wall Street analyst recommendations and price targets from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        data  = {
            "consensus":    info.get("recommendationKey", "N/A").upper(),
            "mean_score":   info.get("recommendationMean"),
            "num_analysts": info.get("numberOfAnalystOpinions"),
            "target_mean":  info.get("targetMeanPrice"),
            "target_high":  info.get("targetHighPrice"),
            "target_low":   info.get("targetLowPrice"),
            "current_price":info.get("currentPrice"),
            "upside_pct":   round(
                ((info.get("targetMeanPrice", 0) - info.get("currentPrice", 1)) /
                  info.get("currentPrice", 1)) * 100, 2
            ) if info.get("targetMeanPrice") and info.get("currentPrice") else None,
        }
        return json.dumps(data)
    except Exception as e:
        return f"ERROR:{str(e)}"
