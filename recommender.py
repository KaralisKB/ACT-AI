from crewai.agent import Agent 
import requests


GROQ_API_KEY = "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"

class RecommenderAgent(Agent):

    def __init__(self):
        super().__init__(
            role="Recommender",
            goal="Provide stock recommendations (Buy, Hold, Sell) based on financial data.",
            backstory="An AI agent that combines financial insights and market trends to give investment advice."
        )


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
        # Define the Groq API endpoint (replace with the actual endpoint URL)
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "max_tokens": 300,  # Adjust based on Groq's requirements
            "temperature": 0.7,
        }

        try:
            # Send the POST request to the Groq API
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)

            # Raise an exception if the API call fails
            response.raise_for_status()

            # Parse and return the response JSON
            return response.json()
        except requests.exceptions.RequestException as e:
            # Handle any HTTP errors or connection issues
            raise Exception(f"Failed to query Groq API: {str(e)}")