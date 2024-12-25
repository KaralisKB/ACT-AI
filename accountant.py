from transformers import pipeline
import math

class AccountantAgent:
    def __init__(self):
        # Load DistilGPT-2 model for reasoning and explanations
        self.ai_model = pipeline("text-generation", model="distilgpt2")
    
    def calculate_ratios(self, financial_data):
        """
        Perform comprehensive accounting calculations using Python.
        """
        try:
            # Extract necessary fields from financial data
            current_price = financial_data["current_price"]
            previous_close = financial_data.get("previous_close", 0)
            eps = financial_data.get("eps", 0)  # Earnings per share
            market_cap = financial_data.get("market_cap", 0)
            dividend_yield = financial_data.get("dividend_yield", 0)
            pe_ratio = financial_data.get("pe_ratio", None)
            open_price = financial_data.get("open_price", 0)
            high_price = financial_data.get("high_price", 0)
            low_price = financial_data.get("low_price", 0)
            week_52_high = financial_data.get("52_week_high", 0)
            week_52_low = financial_data.get("52_week_low", 0)

            # Calculations
            if eps > 0 and not pe_ratio:
                pe_ratio = round(current_price / eps, 2)  # Price-to-Earnings ratio
            
            dividend_percentage = round(dividend_yield * 100, 2) if dividend_yield else None
            price_change = round(current_price - previous_close, 2)  # Absolute price change
            price_change_percent = (
                round((price_change / previous_close) * 100, 2) if previous_close > 0 else None
            )
            volatility = round(high_price - low_price, 2)  # Daily price range
            price_to_book = (
                round(market_cap / (eps * 1_000_000), 2) if eps > 0 else None  # Simplified price-to-book ratio
            )
            price_vs_52_high = round(((week_52_high - current_price) / week_52_high) * 100, 2) if week_52_high > 0 else None
            price_vs_52_low = round(((current_price - week_52_low) / week_52_low) * 100, 2) if week_52_low > 0 else None

            return {
                "pe_ratio": pe_ratio,
                "dividend_yield_percent": dividend_percentage,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "volatility": volatility,
                "price_to_book_ratio": price_to_book,
                "price_vs_52_week_high_percent": price_vs_52_high,
                "price_vs_52_week_low_percent": price_vs_52_low,
            }
        except KeyError as e:
            raise ValueError(f"Missing required financial data field: {str(e)}")

    def ai_insights(self, financial_data, ratios):
        """
        Use DistilGPT-2 to generate textual insights based on financial calculations.
        """
        prompt = (
            f"Analyze the following financial metrics: "
            f"PE Ratio: {ratios['pe_ratio']}, Dividend Yield: {ratios['dividend_yield_percent']}%, "
            f"Price Change: {ratios['price_change']} ({ratios['price_change_percent']}%), "
            f"Volatility: {ratios['volatility']}, Price-to-Book Ratio: {ratios['price_to_book_ratio']}, "
            f"Price vs 52-Week High: {ratios['price_vs_52_week_high_percent']}%, "
            f"Price vs 52-Week Low: {ratios['price_vs_52_week_low_percent']}%. "
            f"Evaluate if this stock is suitable for investment."
        )
        
        # Generate textual insights
        ai_response = self.ai_model(prompt, max_length=150, num_return_sequences=1)
        return ai_response[0]["generated_text"]

    def handle_task(self, task_input):
        """
        Main function to handle incoming tasks.
        """
        financial_data = task_input.get("financial_data", {})
        if not financial_data:
            return {"error": "Financial data is required."}

        # Step 1: Perform calculations using Python
        try:
            ratios = self.calculate_ratios(financial_data)
        except ValueError as e:
            return {"error": str(e)}
        
        # Step 2: Generate insights using DistilGPT-2
        ai_generated_insights = self.ai_insights(financial_data, ratios)

        # Step 3: Combine results into a unified response
        return {
            "ratios": ratios,
            "ai_insights": ai_generated_insights
        }
