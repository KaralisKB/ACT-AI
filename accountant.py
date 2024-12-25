from crewai.agent import Agent
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import os
import json

class AccountantAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Accountant",
            goal="Perform advanced financial calculations and provide insights using AI.",
            backstory="An AI financial analyst powered by GPT-2 for stock-related computations."
        )
        # Initialize GPT-2 model outside the strict attribute system
        self._initialize_model()

    def _initialize_model(self):
        """
        Load GPT-2 model and tokenizer.
        """
        self.generator = pipeline(
            "text-generation",
            model=GPT2LMHeadModel.from_pretrained("gpt2"),
            tokenizer=GPT2Tokenizer.from_pretrained("gpt2"),
        )

    def handle_task(self, researcher_data):
        try:
            financials = researcher_data.get("financial_data", {})
            stock_ticker = researcher_data.get("stock_ticker", "Unknown Stock")

            if not financials:
                return {"error": "No financial data provided by the researcher."}

            # Prepare the input prompt for GPT-2
            prompt = self.construct_prompt(stock_ticker, financials)

            # Use GPT-2 to generate results
            results = self.generate_analysis(prompt)

            # Process the results (e.g., clean up text output)
            processed_results = self.process_results(results)

            return {
                "stock_ticker": stock_ticker,
                "analysis": processed_results
            }
        except Exception as e:
            return {"error": f"Error processing AccountantAgent task: {str(e)}"}

    def construct_prompt(self, stock_ticker, financials):
        """
        Construct the input prompt for GPT-2 based on the financial data.
        """
        financial_data_summary = json.dumps(financials, indent=2)
        prompt = (
            f"You are an AI accountant analyzing financial data for the stock {stock_ticker}. "
            "Using the provided financial data, calculate the following:\n"
            "1. Profitability Ratios: Gross Margin, Operating Margin, Net Margin.\n"
            "2. Liquidity Ratios: Current Ratio, Quick Ratio.\n"
            "3. Risk Metrics: Debt-to-Equity Ratio, Beta.\n"
            "4. Provide a short textual analysis of the company's financial health.\n\n"
            f"Financial Data:\n{financial_data_summary}\n\n"
            "Your analysis:"
        )
        return prompt

    def generate_analysis(self, prompt):
        """
        Use GPT-2 to generate financial analysis based on the input prompt.
        """
        try:
            generated = self.generator(
                prompt,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            )
            return generated[0]["generated_text"]
        except Exception as e:
            raise RuntimeError(f"Error generating analysis with GPT-2: {str(e)}")

    def process_results(self, results):
        """
        Process and format the GPT-2 output for readability.
        """
        # Extract and clean up the GPT-2-generated output
        if isinstance(results, str):
            return results.strip()
        return "No valid analysis generated."
