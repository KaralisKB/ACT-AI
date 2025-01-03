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
            object.__setattr__(self, 'client', Groq(api_key=GROQ_API_KEY))  # Explicitly bypass Pydantic
            logger.debug(f"Groq client initialized with API key: {GROQ_API_KEY[:6]}******")
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

            # Query Groq and process the response
            response = self.query_groq(prompt)
            if not response or "error" in response:
                return {"error": f"Groq API Error: {response.get('error', 'Unknown error')}"}

            # Ensure the recommendation and reasoning are consistent
            recommendation = response.get("recommendation", "").strip()
            reasoning = response.get("reasoning", "").strip()

            # Validate alignment between recommendation and reasoning
            if "hold" in reasoning.lower() and recommendation.lower() != "hold":
                recommendation = "Hold"
            elif "buy" in reasoning.lower() and recommendation.lower() != "buy":
                recommendation = "Buy"
            elif "sell" in reasoning.lower() and recommendation.lower() != "sell":
                recommendation = "Sell"

            # Return consistent results
            return {"recommendation": recommendation, "rationale": reasoning}
        except Exception as e:
            return {"error": f"Recommender Agent Error: {str(e)}"}


    def build_prompt(self, financial_data, calculations, news_articles):
        try:
            news_summary = "\n".join(
                [f"- {article['headline']} (Source: {article['source']}): {article['summary']}" for article in news_articles[:3]]
            )

            prompt = f"""
            You are a financial analyst. Based on the following data, provide a single **consistent** recommendation (Buy, Hold, Sell) and reasoning. The recommendation and reasoning must match.

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
            Based on the above data, provide:
            1. A single recommendation: **Buy**, **Hold**, or **Sell**.
            2. Consistent reasoning supporting this recommendation. Ensure the reasoning and recommendation match.
            """
            logger.debug(f"Built prompt: {prompt[:500]}...")  # Log the prompt for debugging
            return prompt
        except Exception as e:
            logger.error(f"Error building prompt: {str(e)}")
            raise


    def query_groq(self, prompt, temperature=1.6):
        """
        Queries the Groq API to get a recommendation.

        Args:
            prompt (str): The input prompt for the model.
            temperature (float): Sampling temperature, between 0 and 2. Higher values increase randomness.

        Returns:
            dict: The response containing recommendation and reasoning or an error.
        """
        try:
            logger.debug("Sending request to Groq API.")
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=temperature  # Add temperature parameter
            )
            logger.debug(f"Groq API response: {chat_completion}")
            response_message = chat_completion.choices[0].message.content
            return {"recommendation": "Buy", "reasoning": response_message}
        except Exception as e:
            logger.error(f"Failed to query Groq API: {str(e)}")
            raise

