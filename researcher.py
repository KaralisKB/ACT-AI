from crewai.agent import Agent
import openai
import os
import requests

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class ResearcherAgent(Agent):
    def __init__(self):
        # Define the required fields for the parent Agent class
        super().__init__(
            role="Researcher",  # The role of the agent
            goal="Research and analyze stock-related data and news",  # The goal of the agent
            backstory="An AI agent designed to assist with financial research."  # Backstory for the agent
        )

    def handle_task(self, task_input):
        stock_symbol = task_input.get("stock_symbol", "")

        if not stock_symbol:
            return "Error: No stock symbol provided."

        # Fetch stock data using Finnhub API
        try:
            finn_token = os.getenv("FINNHUB_API_KEY")
            finnhub_url = f"https://finnhub.io/api/v1/quote?symbol={stock_symbol}&token={finn_token}"
            response = requests.get(finnhub_url)
            stock_data = response.json()

            # Format the stock data
            stock_info = f"Stock: {stock_symbol}\n"
            stock_info += f"Current Price: {stock_data.get('c', 'N/A')}\n"
            stock_info += f"High: {stock_data.get('h', 'N/A')}\n"
            stock_info += f"Low: {stock_data.get('l', 'N/A')}\n"
            stock_info += f"Open: {stock_data.get('o', 'N/A')}\n"
            stock_info += f"Previous Close: {stock_data.get('pc', 'N/A')}\n"

            return stock_info
        except Exception as e:
            return f"Error fetching stock data: {str(e)}"

    def analyze_news(self, stock_symbol):
        # Simulated function for gathering news
        return f"News for {stock_symbol}: Simulated news content."

# Example usage
#if __name__ == "__main__":
 #   researcher = ResearcherAgent()
  #  task_input = {"stock_symbol": "AAPL"}
   # print(researcher.handle_task(task_input))
