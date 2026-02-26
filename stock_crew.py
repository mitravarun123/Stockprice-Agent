from crewai import Crew 
import os 
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

from agents import (
    TechnicalTask,
    NewsTask,
    TickerTask,
    RiskTask,
    AdvisorTask,
    ReportTask,
    PriceTask,
    AnalystRatingTask
)

ticker_agent          = TickerAgent().agent
price_agent           = PriceAgent().agent
news_agent            = NewsAgent().agent
technical_agent       = TechnicalAgent().agent
analyst_ratings_agent = AnalystRatingsAgent().agent
risk_agent            = RiskAgent().agent
advisor_agent         = AdvisorAgent().agent
report_agent          = ReportAgent().agent

technical_task       = TechnicalTask() 
NewsTask             = NewsTask()
RiskTask             = RiskTask() 
AdvisorTask          = AdvisorTask()
ReportTask           = ReportTask()
PriceTask            = PriceTask()
AnalystRatingTask    = AnalystRatingTask()



stock_crew = Crew(
    agents=[
        ticker_agent,
        price_agent,
        news_agent,
        technical_agent,
        analyst_ratings_agent,
        risk_agent,
        advisor_agent,
        report_agent,
    ],
    tasks=[
        TickerTask,
        PriceTask,
        NewsTask,
        technical_task,
        AnalystRatingTask,
        RiskTask,
        AdvisorTask,
        ReportTask,
    ],
    process=Process.sequential,
    verbose=True
)

