import warnings 
import os 
from crewai import Agent 
from langchain_anthropic import ChatAnthropic
from tools import fetch_price_tool, fetch_news_tool, fetch_technical_tool, fetch_analyst_ratings_tool

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=1024,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)




class TicknerAgent:
    def __init__(self) -> None:
        self.ticker_agent = Agent(
            role="Ticker Extractor",
            goal="Extract the exact stock ticker symbol from the user input",
            backstory="Expert at identifying stock ticker symbols from natural language. Always returns a clean uppercase ticker like AAPL, TSLA, MSFT.",
            llm=llm,
            verbose=True
        )


class PriceAgent:
    def __init__(self):
        self.price_agent = Agent(
            role = "Expert Market Data Fetcher",
            goal = "Fetch live price, volume , market capacity, P/E ration and 52-week range",
            backstory = "Bloomberg terminal operator with direct market data access via yfinance.",
            llm = llm, 
            verbose = True 

        )


class NewsAgent:
    def __init__(self):
        self.news_agent = Agent(
            role="Financial News Researcher",
            goal="Gather latest news headlines and identify overall sentiment for the stock",
            backstory="Financial journalist scanning Yahoo Finance and Finnhub for the most relevant stories.",
            llm=llm,
            tools=[fetch_news_tool],
            verbose=True
        )
    

class TechnicalAgent:
    def __init__(self):
        self.technical_agent = Agent(
                role="Technical Analyst",
                goal="Calculate and interpret RSI, MACD, Bollinger Bands, SMA crossovers and ATR",
                backstory="20-year veteran technical analyst. Expert at reading chart patterns and extracting actionable signals.",
                llm=llm,
                tools=[fetch_technical_tool],
                verbose=True
        )



class AnalystRatingAgent:
    def __init__(self):
        self.analyst_rating_agent = Agent(
            role="Analyst Ratings Researcher",
            goal="Fetch Wall Street consensus ratings, price targets and upside potential",
            backstory="Sell-side research coordinator aggregating analyst opinions from major investment banks.",
            llm=llm,
            tools=[fetch_analyst_ratings_tool],
            verbose=True
        )




class RiskAgent:
    def __init__(self):
        self.risk_agent = Agent(
            role="Risk Analyst",
            goal="Assess overall investment risk on a scale of 1-10 and identify top 3 risk factors",
            backstory="Chief Risk Officer with 15 years at top hedge funds. Evaluates technical, fundamental and sentiment risks.",
            llm=llm,
            verbose=True
        )

class AdvisorAgent:
    def __init__(self):
        self.advisor_agent = Agent(
            role="Investment Advisor",
            goal="Generate a clear BUY, HOLD or SELL recommendation with entry, target and stop loss",
            backstory="Senior portfolio manager overseeing a $2B fund. Makes decisive data-driven investment calls.",
            llm=llm,
            verbose=True
        )

class ReportAgent:
    def __init__(self):
        self.report_agent = Agent(
             role="Research Report Writer",
            goal="Compile all analysis into a clean professional investment research report",
            backstory="CFA-certified financial writer producing clear structured reports that turn complex data into insights.",
            llm=llm,
            verbose=True
        )

