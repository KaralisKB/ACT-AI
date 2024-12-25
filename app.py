from flask import Flask, jsonify, request
from researcher import ResearcherAgent
import requests
import os

app = Flask(__name__)

# Initialize ResearcherAgent
researcher_agent = ResearcherAgent()

# Load the ngrok URL for the local Accountant agent
ACCOUNTANT_NGROK_URL = os.getenv("ACCOUNTANT_NGROK_URL")  # Ensure this environment variable is set

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

        # Step 1: Use the ResearcherAgent to fetch financial data and news
        researcher_result = researcher_agent.handle_task({"stock_ticker": stock_ticker})
        if "error" in researcher_result:
            return jsonify({"error": f"Researcher Agent Error: {researcher_result['error']}"}), 500

        print("Payload sent to Accountant Agent:", researcher_result["financial_data"])

        # Step 2: Send ResearcherAgent result to local AccountantAgent via ngrok
        accountant_url = f"{ACCOUNTANT_NGROK_URL}/accountant"
        try:
            accountant_response = requests.post(accountant_url, json={"financial_data": researcher_result})
            if accountant_response.status_code != 200:
                return jsonify({"error": f"Accountant Agent Error: {accountant_response.text}"}), 500

            accountant_result = accountant_response.json()
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Failed to reach Accountant Agent: {str(e)}"}), 500

        # Combine results from both agents
        combined_result = {
            "researcher_data": researcher_result,
            "accountant_analysis": accountant_result
        }

        return jsonify(combined_result)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
