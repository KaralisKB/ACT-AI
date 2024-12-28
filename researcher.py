from crewai.agent import Agent
import openai
import requests
import os

# Set API keys
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

class ResearcherAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Researcher",
            goal="Research financial data and news about a stock, and analyze insights using OpenAI.",
            backstory="Designed to gather stock-related insights and enhance analysis with LLM capabilities."
        )

    def handle_task(self, task_input):
        stock_ticker = task_input.get("stock_ticker", "")
        if not stock_ticker:
            return {"error": "No stock ticker provided."}

        try:
            # Step 1: Fetch stock data and news from Finnhub
            stock_data = self.fetch_stock_data(stock_ticker)
            news = self.fetch_stock_news(stock_ticker)

            # Step 2: Use OpenAI to analyze the combined data
            openai_analysis = self.analyze_with_openai(stock_data, news)

            return {
                "stock_ticker": stock_ticker,
                "financial_data": stock_data,
                "news": news,
                "openai_analysis": openai_analysis
            }
        except Exception as e:
            return {"error": f"Researcher error: {str(e)}"}

    def fetch_stock_data(self, stock_ticker):
        """
        Fetch financial data about the stock, including price and metrics.
        """
        try:
            # Fetch basic quote data
            quote_url = f"https://finnhub.io/api/v1/quote?symbol={stock_ticker}&token={FINNHUB_API_KEY}"
            quote_response = requests.get(quote_url)
            quote_response.raise_for_status()
            quote_data = quote_response.json()

            # Fetch company profile
            profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={stock_ticker}&token={FINNHUB_API_KEY}"
            profile_response = requests.get(profile_url)
            profile_response.raise_for_status()
            profile_data = profile_response.json()

            # Fetch key financial metrics
            metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={stock_ticker}&metric=all&token={FINNHUB_API_KEY}"
            metrics_response = requests.get(metrics_url)
            metrics_response.raise_for_status()
            metrics_data = metrics_response.json().get("metric", {})

            # Consolidate data
            stock_data = {
                "current_price": quote_data.get("c", "N/A"),
                "open_price": quote_data.get("o", "N/A"),
                "high_price": quote_data.get("h", "N/A"),
                "low_price": quote_data.get("l", "N/A"),
                "previous_close": quote_data.get("pc", "N/A"),
                "market_cap": metrics_data.get("marketCapitalization", "N/A"),
                "pe_ratio": metrics_data.get("peBasicExclExtraTTM", "N/A"),
                "eps": metrics_data.get("epsTTM", "N/A"),
                "dividend_yield": metrics_data.get("dividendYieldIndicatedAnnual", "N/A"),
                "52_week_high": metrics_data.get("52WeekHigh", "N/A"),
                "52_week_low": metrics_data.get("52WeekLow", "N/A"),
                "company_name": profile_data.get("name", "N/A"),
                "industry": profile_data.get("finnhubIndustry", "N/A"),
                "website": profile_data.get("weburl", "N/A"),
            }
            return stock_data
        except Exception as e:
            raise Exception(f"Failed to fetch stock data: {str(e)}")

    def fetch_stock_news(self, stock_ticker):
        """
        Fetch the latest news articles about the stock.
        """
        try:
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={stock_ticker}&from=2023-01-01&to=2024-12-31&token={FINNHUB_API_KEY}"
            news_response = requests.get(news_url)
            news_response.raise_for_status()
            news_data = news_response.json()

            # Extract and format news items
            news_items = []
            for item in news_data[:5]:  # Limit to the top 5 news items
                news_items.append({
                    "headline": item.get("headline", "N/A"),
                    "source": item.get("source", "N/A"),
                    "datetime": item.get("datetime", "N/A"),
                    "summary": item.get("summary", "N/A"),
                    "url": item.get("url", "N/A"),
                })
            return news_items
        except Exception as e:
            raise Exception(f"Failed to fetch stock news: {str(e)}")

    def analyze_with_openai(self, stock_data, news):
        """
        Use OpenAI to provide insights on stock data and news.
        """
        try:
            # Summarize news for analysis
            news_summary = "\n".join(
                [f"- {item['headline']} (Source: {item['source']}): {item['summary']}" for item in news]
            )

            # Build a detailed prompt
            prompt = f"""
            Analyze the following stock data and news to provide insights:

            **Financial Data**:
            - Current Price: {stock_data.get('current_price', 'N/A')}
            - 52-Week High: {stock_data.get('52_week_high', 'N/A')}
            - 52-Week Low: {stock_data.get('52_week_low', 'N/A')}
            - Market Cap: {stock_data.get('market_cap', 'N/A')}
            - PE Ratio: {stock_data.get('pe_ratio', 'N/A')}
            - EPS: {stock_data.get('eps', 'N/A')}
            - Dividend Yield: {stock_data.get('dividend_yield', 'N/A')}

            **Recent News**:
            {news_summary}

            **Task**:
            Based on the above data, summarize the key insights and trends, and explain whether the stock shows potential for growth or risk.
            """

            # Call OpenAI's GPT
            response = openai.Completion.create(
                engine="gpt-4",
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].text.strip()
        except Exception as e:
            raise Exception(f"Failed to analyze with OpenAI: {str(e)}")
