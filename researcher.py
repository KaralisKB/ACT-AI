from crewai.agent import Agent
import openai
import os
import requests

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

class ResearcherAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Researcher",
            goal="Research and analyze all financial data and news about a given stock.",
            backstory="An AI agent designed to provide detailed stock research, including financials and news."
        )

    def handle_task(self, task_input):
        stock_ticker = task_input.get("stock_ticker", "")

        if not stock_ticker:
            return "Error: No stock symbol provided."

        try:
            # Fetch stock data and news from Finnhub
            stock_data = self.fetch_stock_data(stock_ticker)
            news = self.fetch_stock_news(stock_ticker)

            # Combine results into a response
            response = {
                "stock_ticker": stock_ticker,
                "financial_data": stock_data,
                "news": news,
            }
            return response
        except Exception as e:
            return f"Error processing the stock research: {str(e)}"

    def fetch_stock_data(self, stock_ticker):
        """
        Fetch financial data about the stock, including:
        - Current price, open, high, low, previous close
        - Key metrics (PE ratio, market cap, EPS, etc.)
        """
        try:
            # Fetch basic quote data
            quote_url = f"https://finnhub.io/api/v1/quote?symbol={stock_ticker}&token={FINNHUB_API_KEY}"
            quote_response = requests.get(quote_url)
            quote_response.raise_for_status()
            quote_data = quote_response.json()

            # Fetch company profile
            profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={stock_ticker}&token={FINNHUB_API_KEY}"
            profile_response = requests.get(profile_url)
            profile_response.raise_for_status()
            profile_data = profile_response.json()

            # Fetch key financial metrics
            metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={stock_ticker}&metric=all&token={FINNHUB_API_KEY}"
            metrics_response = requests.get(metrics_url)
            metrics_response.raise_for_status()
            metrics_data = metrics_response.json().get("metric", {})

            # Consolidate data
            stock_data = {
                "current_price": quote_data.get("c", "N/A"),
                "open_price": quote_data.get("o", "N/A"),
                "high_price": quote_data.get("h", "N/A"),
                "low_price": quote_data.get("l", "N/A"),
                "previous_close": quote_data.get("pc", "N/A"),
                "market_cap": metrics_data.get("marketCapitalization", "N/A"),
                "pe_ratio": metrics_data.get("peBasicExclExtraTTM", "N/A"),
                "eps": metrics_data.get("epsTTM", "N/A"),
                "dividend_yield": metrics_data.get("dividendYieldIndicatedAnnual", "N/A"),
                "52_week_high": metrics_data.get("52WeekHigh", "N/A"),
                "52_week_low": metrics_data.get("52WeekLow", "N/A"),
                "company_name": profile_data.get("name", "N/A"),
                "industry": profile_data.get("finnhubIndustry", "N/A"),
                "website": profile_data.get("weburl", "N/A"),
            }
            return stock_data
        except Exception as e:
            raise Exception(f"Failed to fetch stock data: {str(e)}")

    def fetch_stock_news(self, stock_ticker):
        """
        Fetch the latest news articles about the stock.
        """
        try:
            # Fetch news from Finnhub
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={stock_ticker}&from=2023-01-01&to=2024-12-31&token={FINNHUB_API_KEY}"
            news_response = requests.get(news_url)
            news_response.raise_for_status()
            news_data = news_response.json()

            # Extract and format news items
            news_items = []
            for item in news_data[:5]:  # Limit to the top 5 news items
                news_items.append({
                    "headline": item.get("headline", "N/A"),
                    "source": item.get("source", "N/A"),
                    "datetime": item.get("datetime", "N/A"),
                    "summary": item.get("summary", "N/A"),
                    "url": item.get("url", "N/A"),
                })
            return news_items
        except Exception as e:
            raise Exception(f"Failed to fetch stock news: {str(e)}")
