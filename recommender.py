import cohere
import os

class RecommenderAgent:
    def __init__(self):
        # Initialize Cohere with API key
        self.api_key = os.getenv("COHERE_API_KEY")  # Set this environment variable
        self.client = cohere.Client(self.api_key)

    def handle_task(self, researcher_data, accountant_data):
        # Combine data from researcher and accountant
        financial_data = researcher_data.get("financial_data", {})
        calculations = accountant_data.get("calculations", {})
        news_articles = researcher_data.get("news", [])

        # Build the prompt for the LLM
        prompt = self.build_prompt(financial_data, calculations, news_articles)
        print("[DEBUG] Generated Prompt:", prompt)  # Debug prompt

        # Send the request to Cohere
        try:
            response = self.generate_response(prompt)
            return {"recommendation": response}
        except Exception as e:
            return {"error": f"Failed to connect to Cohere API: {str(e)}"}

    def build_prompt(self, financial_data, calculations, news_articles):
        # Craft a detailed and structured prompt
        news_summary = "\n".join(
            [f"- {article['headline']} (Source: {article['source']}): {article['summary']}"
             for article in news_articles[:3]]  # Limit to 3 articles
        )

        prompt = f"""
        Analyze the following stock data and news to determine whether to Buy, Hold, or Sell. Provide clear reasoning and highlight key factors.

        **Financial Data**:
        - Company Name: {financial_data.get('company_name', 'N/A')}
        - Industry: {financial_data.get('industry', 'N/A')}
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

        **Recommendation**:
        Provide a detailed analysis and specify whether the stock is a Buy, Hold, or Sell. Explain your reasoning.
        """
        return prompt

    def generate_response(self, prompt):
        # Send the prompt to Cohere's Generate API
        response = self.client.generate(
            model="command-xlarge",  # Or use "command-medium" for a smaller model
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            k=0,
            p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        print("[DEBUG] Cohere Response:", response.generations[0].text)  # Debug response
        return response.generations[0].text.strip()
