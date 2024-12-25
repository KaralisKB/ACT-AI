import os
import openai

class RecommenderAgent:
    def __init__(self):
        # Set up GooseAI API key and endpoint
        self.api_key = os.getenv("GOOSEAI_API_KEY")  # Ensure this is set in your environment
        self.api_url = "https://api.goose.ai/v1"  # GooseAI's base URL for the OpenAI client
        openai.api_key = self.api_key
        openai.api_base = self.api_url

    def handle_task(self, researcher_data, accountant_data):
        # Combine data from researcher and accountant
        financial_data = researcher_data.get("financial_data", {})
        calculations = accountant_data.get("calculations", {})
        news_articles = researcher_data.get("news", [])

        # Build the prompt for the LLM
        prompt = self.build_prompt(financial_data, calculations, news_articles)
        print("Generated Prompt:", prompt)  # Debugging: Print the prompt to verify its structure

        # Use OpenAI client to generate response
        try:
            response = self.generate_response(prompt)
            recommendation = response.choices[0].text.strip()
            return {"recommendation": recommendation}
        except Exception as e:
            return {"error": f"Failed to generate recommendation: {str(e)}"}

    def build_prompt(self, financial_data, calculations, news_articles):
        # Craft a detailed and structured prompt with data provided
        news_summary = "\n".join(
            [
                f"- {article['headline']} (Source: {article['source']}): {article['summary']}"
                for article in news_articles[:3]  # Limit to 3 articles
            ]
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
        # Use OpenAI client to make the API call
        response = openai.Completion.create(
            engine="fairseq-13b",  # Change to your preferred engine
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response
