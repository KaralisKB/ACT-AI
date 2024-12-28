from flask import Flask, jsonify, request
from researcher import ResearcherAgent
from recommender import RecommenderAgent
import requests
import os

app = Flask(__name__)

# Initialize agents
researcher_agent = ResearcherAgent()
recommender_agent = RecommenderAgent()

# Load Ngrok URLs for local agents
ACCOUNTANT_NGROK_URL = os.getenv("ACCOUNTANT_NGROK_URL")
BLOGGER_NGROK_URL = os.getenv("BLOGGER_NGROK_URL")

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json

        # Validate input
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload."}), 400

        stock_ticker = data.get("stock_ticker", "")
        if not stock_ticker:
            return jsonify({"error": "Stock ticker is required."}), 400

        # Step 1: ResearcherAgent fetches financial data and news
        researcher_result = researcher_agent.handle_task({"stock_ticker": stock_ticker})
        if "error" in researcher_result:
            return jsonify({"error": f"Researcher Agent Error: {researcher_result['error']}"}), 500

        # Step 2: Send Researcher data to AccountantAgent (via Ngrok)
        accountant_url = f"{ACCOUNTANT_NGROK_URL}/accountant"
        try:
            accountant_response = requests.post(
                accountant_url,
                json={"financial_data": researcher_result["financial_data"]},
                timeout=30
            )
            if accountant_response.status_code != 200:
                return jsonify({"error": f"Accountant Agent Error: {accountant_response.text}"}), 500

            accountant_result = accountant_response.json()
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Failed to reach Accountant Agent: {str(e)}"}), 500

        # Step 3: RecommenderAgent generates a recommendation
        recommender_result = recommender_agent.handle_task(researcher_result, accountant_result)
        if "error" in recommender_result:
            return jsonify({"error": f"Recommender Agent Error: {recommender_result['error']}"}), 500

        # Step 4: Send combined results to BloggerAgent (via Ngrok)
        blogger_url = f"{BLOGGER_NGROK_URL}/blogger"
        try:
            blogger_response = requests.post(
                blogger_url,
                json={
                    "research_data": researcher_result,
                    "accountant_analysis": accountant_result,
                    "recommendation": recommender_result.get("recommendation", "No recommendation provided")
                },
                timeout=30
            )
            if blogger_response.status_code != 200:
                return jsonify({"error": f"Blogger Agent Error: {blogger_response.text}"}), 500

            blogger_result = blogger_response.json()
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Failed to reach Blogger Agent: {str(e)}"}), 500

        # Step 5: Combine results from all agents
        combined_result = {
            "researcher_data": researcher_result,
            "accountant_analysis": accountant_result,
            "recommendation": recommender_result.get("recommendation", "No recommendation provided"),
            "blog_report": blogger_result.get("report", "No report generated.")
        }

        return jsonify(combined_result)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
