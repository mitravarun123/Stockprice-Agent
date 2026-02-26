from crewai import Task
from agents import (
    TickerAgent,        
    PriceAgent,
    NewsAgent,
    TechnicalAgent,
    AnalystRatingsAgent,
    RiskAgent,
    AdvisorAgent,
    ReportAgent
)


ticker_agent          = TickerAgent().agent
price_agent           = PriceAgent().agent
news_agent            = NewsAgent().agent
technical_agent       = TechnicalAgent().agent
analyst_ratings_agent = AnalystRatingsAgent().agent
risk_agent            = RiskAgent().agent
advisor_agent         = AdvisorAgent().agent
report_agent          = ReportAgent().agent


class TickerTask:
    def __init__(self):
        self.ticker_task = Task(
            description="""
            Extract the stock ticker symbol from this user input: {user_input}
            Return ONLY the uppercase ticker symbol. Example: AAPL
            If you cannot find a ticker, return UNKNOWN.
            """,
            expected_output="A single uppercase stock ticker symbol like AAPL or UNKNOWN",
            agent=ticker_agent
        )


class PriceTask:
    def __init__(self):
        ticker = TickerTask()                 
        self.price_task = Task(
            description="""
            Use the Fetch Price Tool to get live price data for the ticker.
            Return the full price data including:
            price, change%, PE ratio, market cap, 52W high/low, sector, volume, beta.
            """,
            expected_output="JSON with price, change_pct, pe_ratio, market_cap, 52w_high, 52w_low, sector, volume, beta",
            agent=price_agent,
            context=[ticker.ticker_task]       
        )


class NewsTask:
    def __init__(self):
        ticker = TickerTask()
        self.news_task = Task(
            description="""
            Use the Fetch News Tool to get the latest headlines for the ticker.
            After fetching provide:
            1. The list of headlines
            2. Overall sentiment: POSITIVE, NEGATIVE or NEUTRAL
            3. Top 2 most impactful headlines and why they matter
            """,
            expected_output="Headlines list, sentiment (POSITIVE/NEGATIVE/NEUTRAL), top 2 impactful stories",
            agent=news_agent,
            context=[ticker.ticker_task]      
        )


class TechnicalTask:
    def __init__(self):
        ticker = TickerTask()
        self.technical_task = Task(
            description="""
            Use the Fetch Technical Tool to calculate all indicators for the ticker.
            Interpret the results:
            1. Overall technical trend: BULLISH, BEARISH or NEUTRAL
            2. List all indicator signals
            3. Strongest signal supporting your view
            4. Any conflicting signals
            """,
            expected_output="Technical indicators, trend (BULLISH/BEARISH/NEUTRAL), key signals and conflicts",
            agent=technical_agent,
            context=[ticker.ticker_task]       
        )


class AnalystRatingTask:
    def __init__(self):
        ticker = TickerTask()
        price  = PriceTask()
        self.analyst_ratings_task = Task(   
            description="""
            Use the Fetch Analyst Ratings Tool to get Wall Street consensus data.
            Summarize:
            1. Consensus rating and what it means
            2. Average price target and upside/downside from current price
            3. Range between highest and lowest targets
            4. Number of analysts covering the stock
            """,
            expected_output="Consensus rating, mean price target, upside %, target range, analyst count",
            agent=analyst_ratings_agent,
            context=[ticker.ticker_task, price.price_task]   
        )


class RiskTask:
    def __init__(self):
        price    = PriceTask()
        technical= TechnicalTask()
        news     = NewsTask()
        analyst  = AnalystRatingTask()
        self.risk_task = Task(
            description="""
            Using ALL collected data (price, technical, news, analyst ratings) provide:
            1. Overall risk score: 1 (very safe) to 10 (very risky)
            2. Top 3 specific risk factors with explanation
            3. Top 2 positive factors that reduce risk
            4. Risk category: LOW / MODERATE / HIGH / VERY HIGH
            """,
            expected_output="Risk score 1-10, risk category, top 3 risks, top 2 positives",
            agent=risk_agent,
            context=[
                price.price_task,
                technical.technical_task,
                news.news_task,
                analyst.analyst_ratings_task    
            ]
        )


class AdvisorTask:
    def __init__(self):
        price    = PriceTask()
        technical= TechnicalTask()
        news     = NewsTask()
        risk     = RiskTask()
        analyst  = AnalystRatingTask()
        self.advisor_task = Task(
            description="""
            Generate a clear investment recommendation:
            1. Action: BUY / HOLD / SELL / STRONG BUY / STRONG SELL
            2. Conviction: LOW / MEDIUM / HIGH
            3. Suggested entry price range
            4. Bull case target price with % upside
            5. Base case target price with % upside
            6. Stop loss level with % downside
            7. Suggested position size % of portfolio
            8. Time horizon: Short / Medium / Long term
            9. 3 key reasons for your recommendation
            10. 2 key risks that could invalidate your thesis
            """,
            expected_output="Full recommendation with action, entry, targets, stop loss, position size and reasoning",
            agent=advisor_agent,
            context=[
                price.price_task,
                technical.technical_task,
                news.news_task,
                risk.risk_task,
                analyst.analyst_ratings_task
            ]
        )


class ReportTask:
    def __init__(self):
        price    = PriceTask()
        news     = NewsTask()
        technical= TechnicalTask()
        analyst  = AnalystRatingTask()
        risk     = RiskTask()
        advisor  = AdvisorTask()
        self.report_task = Task(
            description="""
            Compile everything into a professional research report:

            ═══════════════════════════════════════════
            [COMPANY NAME] ([TICKER]) — Research Report
            Generated: [DATE]
            ═══════════════════════════════════════════

            📊 MARKET SNAPSHOT
            📈 TECHNICAL ANALYSIS
            📰 NEWS & SENTIMENT
            🏦 ANALYST CONSENSUS
            ⚠️  RISK ASSESSMENT
            💡 INVESTMENT RECOMMENDATION
            🔑 KEY TAKEAWAYS

            ─────────────────────────────────────────
            Educational purposes only. Not financial advice.
            ─────────────────────────────────────────
            """,
            expected_output="Complete formatted investment research report",
            agent=report_agent,
            context=[
                price.price_task,
                news.news_task,
                technical.technical_task,
                analyst.analyst_ratings_task,
                risk.risk_task,
                advisor.advisor_task
            ]
        )
