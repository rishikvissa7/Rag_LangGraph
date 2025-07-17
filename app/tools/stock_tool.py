import yfinance as yf
from langchain.tools import tool
import dateparser
from datetime import timedelta

@tool
def stock_trend_tool(company: str, start_date: str, end_date: str):
    """
    Get stock price trend (open/close/high/low) over a date range for a given company.

    Dates can be natural language like 'last Monday', '2024-07-01', '3 days ago'.
    """
    ticker_map = {
        "apple": "AAPL",
        "google": "GOOGL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "meta": "META",
    }

    ticker = ticker_map.get(company.lower())
    if not ticker:
        return f"Unknown company: {company}"

    start = dateparser.parse(start_date)
    end = dateparser.parse(end_date)
    if not start or not end:
        return "Invalid dates. Try '2024-07-01 to 2024-07-10' or 'last week to today'."

    data = yf.Ticker(ticker).history(start=start.date(), end=(end + timedelta(days=1)).date())
    if data.empty:
        return f"No data for {company.title()} between {start.date()} and {end.date()}."

    response = f"{company.title()} stock trend from {start.date()} to {end.date()}:\n"
    for date, row in data.iterrows():
        response += f"{date.date()}: Open=${row['Open']:.2f}, Close=${row['Close']:.2f}, High=${row['High']:.2f}, Low=${row['Low']:.2f}\n"

    return response
