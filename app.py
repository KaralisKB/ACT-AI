from flask import Flask, jsonify, request
from researcher import ResearcherAgent
from recommender import RecommenderAgent
from accountant import CrewAIAccountantAgent
from blogger import CrewAIBloggerAgent
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize agents
researcher_agent = ResearcherAgent()
recommender_agent = RecommenderAgent()
ngrok_url = os.getenv("LOCAL_NGROK_URL")  # Shared ngrok URL for local models
accountant_agent = CrewAIAccountantAgent()
blogger_agent = CrewAIBloggerAgent()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json

        if not data:
            return jsonify({"error": "Invalid or missing JSON payload."}), 400

        stock_ticker = data.get("stock_ticker", "")
        if not stock_ticker:
            return jsonify({"error": "Stock ticker is required."}), 400

        # Step 1: Use the Researcher Agent
        researcher_result = researcher_agent.handle_task({"stock_ticker": stock_ticker})
        if "error" in researcher_result:
            return jsonify({"error": f"Researcher Agent Error: {researcher_result['error']}"}), 500

        # Fix missing or invalid fields before sending to Accountant
        financial_data = researcher_result.get("financial_data", {})
        financial_data["dividend_yield"] = (
            financial_data.get("dividend_yield") 
            if financial_data.get("dividend_yield") not in [None, "N/A"] 
            else 0.0  # Default to 0.0 if missing or invalid
        )

        # Step 2: Use the Accountant Agent
        accountant_result = accountant_agent.handle_task({"financial_data": financial_data})
        if "error" in accountant_result:
            return jsonify({"error": f"Accountant Agent Error: {accountant_result['error']}"}), 500

        # Step 3: Use the Recommender Agent
        recommender_result = recommender_agent.handle_task(researcher_result, accountant_result)
        if "error" in recommender_result:
            return jsonify({"error": f"Recommender Agent Error: {recommender_result['error']}"}), 500

        # Step 4: Use the Blogger Agent
        blogger_result = blogger_agent.handle_task({
            "recommendation": recommender_result.get("recommendation"),
            "rationale": recommender_result.get("rationale")
        })

        if "error" in blogger_result:
            return jsonify({"error": f"Blogger Agent Error: {blogger_result['error']}"}), 500

        # Updated key for Blogger Agent's output
        blog_summary = blogger_result.get("blog_post", "No blog post generated.")

        # Combine and send the final response
        combined_result = {
            "researcher_data": researcher_result,
            "accountant_analysis": accountant_result,
            "recommendation": recommender_result.get("recommendation", "No recommendation provided"),
            "blog_post": blog_summary  # Updated key for Blogger's summary
        }

        return jsonify(combined_result)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



if __name__ == "__main__":
    app.run(debug=True)
