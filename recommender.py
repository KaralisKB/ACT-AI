import os
import requests

class RecommenderAgent:
    def __init__(self):
        self.api_key = os.getenv("GOOSEAI_API_KEY")  # Ensure this environment variable is set
        self.api_url = "https://api.goose.ai/v1/engines/gpt-neo-20b/completions"  # Change to your preferred engine

    def handle_task(self, researcher_data, accountant_data):
        # Combine data from researcher and accountant
        financial_data = researcher_data.get("financial_data", {})
        calculations = accountant_data.get("calculations", {})
        news_articles = researcher_data.get("news", [])

        # Build the prompt for the LLM
        prompt = self.build_prompt(financial_data, calculations, news_articles)

        # Send the request to Goose.ai
        try:
            response = self.generate_response(prompt)
            if response.status_code != 200:
                return {"error": f"API Error: {response.text}"}

            recommendation = response.json().get("choices", [{}])[0].get("text", "").strip()
            return {"recommendation": recommendation}
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to connect to Goose.ai: {str(e)}"}

    def build_prompt(self, financial_data, calculations, news_articles):
        # Craft a detailed prompt with data provided
        news_summary = "\n".join(
            [
                f"- {article['headline']} (Source: {article['source']}): {article['summary']}"
                for article in news_articles
            ]
        )

        prompt = f"""
        Based on the following financial data, analysis, and recent news articles, provide a recommendation to buy, sell, or hold the stock. Include reasoning behind the recommendation:

        Financial Data:
        - Company Name: {financial_data.get('company_name', 'N/A')}
        - Industry: {financial_data.get('industry', 'N/A')}
        - Current Price: {financial_data.get('current_price', 'N/A')}
        - 52-Week High: {financial_data.get('52_week_high', 'N/A')}
        - 52-Week Low: {financial_data.get('52_week_low', 'N/A')}
        - Market Cap: {financial_data.get('market_cap', 'N/A')}
        - PE Ratio: {financial_data.get('pe_ratio', 'N/A')}
        - Dividend Yield: {financial_data.get('dividend_yield', 'N/A')}

        Analysis:
        - Price-to-Earnings Ratio: {calculations.get('PE_ratio', 'N/A')}
        - Dividend Payout Ratio: {calculations.get('Dividend_payout_ratio', 'N/A')}
        - Growth Rate: {calculations.get('Growth_rate', 'N/A')}
        - Price-to-Book Ratio: {calculations.get('Price_to_book_ratio', 'N/A')}
        - Debt-to-Equity Ratio: {calculations.get('Debt_to_equity_ratio', 'N/A')}

        Recent News Articles:
        {news_summary}

        Recommendation:
        """
        return prompt

    def generate_response(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        response = requests.post(self.api_url, json=payload, headers=headers)
        return response
