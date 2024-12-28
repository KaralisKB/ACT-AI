from crewai.agent import Agent
from groq import Groq
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global API key for Groq
GROQ_API_KEY = "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"

class RecommenderAgent(Agent):
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like Groq

    def __init__(self):
        super().__init__(
            role="Recommender",
            goal="Provide stock recommendations (Buy, Hold, Sell) based on financial data.",
            backstory="An AI agent that combines financial insights and market trends to give investment advice."
        )
        try:
            object.__setattr__(self, 'client', Groq(api_key=GROQ_API_KEY))  # Explicitly set client bypassing Pydantic
            logger.debug("Groq client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            raise

    def handle_task(self, researcher_data, accountant_data):
        try:
            # Combine inputs from Researcher and Accountant agents
            financial_data = researcher_data.get("financial_data", {})
            calculations = accountant_data.get("calculations", {})
            news_articles = researcher_data.get("news", [])

            # Build the input prompt for Groq
            prompt = self.build_prompt(financial_data, calculations, news_articles)

            # Send the request to Groq and process the response
            response = self.query_groq(prompt)
            if not response or "recommendation" not in response or "reasoning" not in response:
                return {"error": "Groq API Error: Missing recommendation or reasoning in response."}

            # Extract and return the recommendation
            recommendation = response["recommendation"]
            rationale = response["reasoning"]

            # Ensure both fields are passed to the Blogger Agent
            return {"recommendation": recommendation, "rationale": rationale}
        except Exception as e:
            return {"error": f"Recommender Agent Error: {str(e)}"}


    def build_prompt(self, financial_data, calculations, news_articles):
        """
        Create a detailed prompt using financial and calculated data along with news summaries.
        """
        try:
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
        except Exception as e:
            logger.error(f"Error building prompt: {str(e)}")
            raise

    def query_groq(self, prompt):
        try:
            logging.debug("Sending request to Groq API.")
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            logging.debug(f"Groq API response: {chat_completion}")
            response_message = chat_completion.choices[0].message.content
            return {"recommendation": "Buy", "reasoning": response_message}
        except Exception as e:
            logging.error(f"Failed to query Groq API: {str(e)}")
            return {"error": f"Failed to query Groq API: {str(e)}"}
