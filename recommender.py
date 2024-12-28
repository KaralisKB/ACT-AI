import os
from crewai.agent import Agent

class RecommenderAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Recommender",
            goal="Analyze data and provide a Buy, Hold, or Sell recommendation.",
            backstory="An AI agent designed to analyze financial and market data to assist investors in decision-making."
        )
        # Initialize Groq API (replace with actual initialization if applicable)
        self.groq_api_key = os.getenv("GROQ_API_KEY")

    def handle_task(self, researcher_data, accountant_data):
        try:
            # Combine inputs from Researcher and Accountant agents
            financial_data = researcher_data.get("financial_data", {})
            calculations = accountant_data.get("calculations", {})
            news_articles = researcher_data.get("news", [])

            # Build the input prompt for Groq
            prompt = self.build_prompt(financial_data, calculations, news_articles)

            # Send the request to Groq (replace with actual Groq API call)
            response = self.query_groq(prompt)
            if "error" in response:
                return {"error": f"Groq API Error: {response['error']}"}

            # Extract and return the recommendation
            recommendation = response.get("recommendation", "No recommendation provided")
            return {"recommendation": recommendation}
        except Exception as e:
            return {"error": f"Recommender Agent Error: {str(e)}"}

    def build_prompt(self, financial_data, calculations, news_articles):
        """
        Create a detailed prompt using financial and calculated data along with news summaries.
        """
        news_summary = "\n".join(
            [f"- {article['headline']} (Source: {article['source']}): {article['summary']}" for article in news_articles[:3]]
        )

        prompt = f"""
        Analyze the following financial data, calculations, and news to determine whether the stock is a Buy, Hold, or Sell:

        **Financial Data**:
        - Current Price: {financial_data.get('current_price', 'N/A')}
        - 52-Week High: {financial_data.get('52_week_high', 'N/A')}
        - 52-Week Low: {financial_data.get('52_week_low', 'N/A')}
        - Market Cap: {financial_data.get('market_cap', 'N/A')}
        - PE Ratio: {financial_data.get('pe_ratio', 'N/A')}
        - Dividend Yield: {financial_data.get('dividend_yield', 'N/A')}

        **Analysis**:
        - Price-to-Earnings Ratio: {calculations.get('PE_ratio', 'N/A')}
        - Dividend Payout Ratio: {calculations.get('Dividend_payout_ratio', 'N/A')}
        - Growth Rate: {calculations.get('Growth_rate', 'N/A')}
        - Price-to-Book Ratio: {calculations.get('Price_to_book_ratio', 'N/A')}
        - Debt-to-Equity Ratio: {calculations.get('Debt_to_equity_ratio', 'N/A')}

        **Recent News**:
        {news_summary}

        **Task**:
        Based on the above data, provide a clear recommendation (Buy, Hold, or Sell) and explain your reasoning.
        """
        return prompt

    def query_groq(self, prompt):
        """
        Query Groq API to generate a recommendation based on the provided prompt.
        """
        try:
            # Example Groq API integration (replace with actual API call)
            response = {
                "recommendation": "Buy",
                "reasoning": "The stock shows strong financial health, high growth potential, and favorable market trends."
            }
            return response
        except Exception as e:
            return {"error": f"Failed to query Groq API: {str(e)}"}
