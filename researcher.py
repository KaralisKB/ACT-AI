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
        stock_ticker = task_input.get("stock_ticker", "")

        if not stock_ticker:
            return "Error: No stock symbol provided."

        # Example research logic (you can replace this with your actual logic)
        try:
            # Fetch stock data from Finnhub
            stock_data = self.fetch_stock_data(stock_ticker)

        # Simulate generating a response (modify as needed)
            response = {
                "stock_ticker": stock_ticker,
                "current_price": stock_data.get("current_price", "Unknown"),
                "news": stock_data.get("news", [])
            }
            return response
        except Exception as e:
            return f"Error processing the stock research: {str(e)}"


    def analyze_news(self, stock_symbol):
        # Simulated function for gathering news
        return f"News for {stock_symbol}: Simulated news content."

# Example usage
#if __name__ == "__main__":
 #   researcher = ResearcherAgent()
  #  task_input = {"stock_symbol": "AAPL"}
   # print(researcher.handle_task(task_input))
