import os
import openai
from crewai.agent import Agent
import requests

# Set OpenAI API Key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

class ResearcherAgent(Agent):
    def __init__(self):
        super().__init__("ResearcherAgent", "An AI researcher agent to analyze stock data.")

    def handle_task(self, task_input):
        stock_ticker = task_input.get("stock_ticker", "")

        # Validate input
        if not stock_ticker:
            return "Error: Stock ticker is required."

        # Fetch Finnhub API Key from environment
        finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        if not finnhub_api_key:
            return "Error: Finnhub API Key is missing in environment variables."

        # Fetch stock data
        stock_data = self.fetch_stock_data(stock_ticker, finnhub_api_key)
        if not stock_data:
            return f"Error: Failed to fetch data for stock {stock_ticker}."

        # Generate stock news
        stock_news = self.fetch_stock_news(stock_ticker)

        # Combine the results
        return {
            "stock_data": stock_data,
            "news": stock_news,
        }

    def fetch_stock_data(self, stock_ticker, api_key):
        url = f"https://finnhub.io/api/v1/quote?symbol={stock_ticker}&token={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            return {
                "current_price": data.get("c"),
                "high_price": data.get("h"),
                "low_price": data.get("l"),
                "open_price": data.get("o"),
                "previous_close": data.get("pc"),
            }
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None

    def fetch_stock_news(self, stock_ticker):
        # Example: Use OpenAI to "scour" the web
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst."},
                    {"role": "user", "content": f"Find recent news and events about the stock {stock_ticker}."},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error fetching stock news: {e}")
            return "Unable to fetch news at the moment."
