from transformers import AutoModelForCausalLM, AutoTokenizer
from crewai.agent import Agent
import torch


# Load model and tokenizer globally (caches the model at startup)
MODEL_NAME = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class AccountantAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Accountant",
            goal="Analyze financial data and generate accounting insights.",
            backstory="A financial accountant AI assistant."
        )
        # Use the preloaded model and tokenizer
        self.model = model
        self.tokenizer = tokenizer

    def handle_task(self, task_input):
        try:
            # Extract financial data from the task input
            financial_data = task_input.get("financial_data", {})
            if not financial_data:
                return {"error": "No financial data provided for analysis."}

            # Perform calculations
            calculations = self.perform_calculations(financial_data)

            # Generate AI-based analysis
            ai_analysis = self.generate_ai_analysis(financial_data)

            # Combine calculations and AI analysis
            response = {
                "calculations": calculations,
                "ai_analysis": ai_analysis,
            }
            return response

        except Exception as e:
            return {"error": f"An error occurred during analysis: {str(e)}"}

    def perform_calculations(self, financial_data):
        # Extract necessary data
        current_price = financial_data.get("current_price", 0)
        eps = financial_data.get("eps", 0)
        pe_ratio = financial_data.get("pe_ratio", 0)
        dividend_yield = financial_data.get("dividend_yield", 0)
        previous_close = financial_data.get("previous_close", 0)
        market_cap = financial_data.get("market_cap", 0)
        week_52_high = financial_data.get("52_week_high", 0)
        week_52_low = financial_data.get("52_week_low", 0)

        # Perform calculations
        calculations = {
            "price_change": current_price - previous_close,
            "price_to_book_ratio": (
                market_cap / eps if eps > 0 else "N/A"
            ),
            "dividend_payout_ratio": (
                dividend_yield * 100 if dividend_yield > 0 else "N/A"
            ),
            "volatility": week_52_high - week_52_low,
            "upside_potential": week_52_high - current_price,
            "downside_risk": current_price - week_52_low,
        }

        return calculations

    def generate_ai_analysis(self, financial_data):
        # Prepare input text for GPT-2
        input_text = (
            f"Company Name: {financial_data.get('company_name', 'N/A')}\n"
            f"Industry: {financial_data.get('industry', 'N/A')}\n"
            f"Current Price: {financial_data.get('current_price', 'N/A')}\n"
            f"Market Cap: {financial_data.get('market_cap', 'N/A')}\n"
            f"52-Week High: {financial_data.get('52_week_high', 'N/A')}\n"
            f"52-Week Low: {financial_data.get('52_week_low', 'N/A')}\n"
            f"Provide a brief accounting analysis of the stock."
        )

        # Tokenize input
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate AI-based text
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the generated text
        ai_analysis = tokenizer.decode(output[0], skip_special_tokens=True)
        return ai_analysis
