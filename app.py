from flask import Flask, jsonify, request
from researcher import ResearcherAgent
from accountant import AccountantAgent

app = Flask(__name__)

# Initialize agents
researcher_agent = ResearcherAgent()
accountant_agent = AccountantAgent()

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

        # Step 2: Use the AccountantAgent to calculate ratios and generate insights
        accountant_result = accountant_agent.handle_task(researcher_result)
        if "error" in accountant_result:
            return jsonify({"error": f"Accountant Agent Error: {accountant_result['error']}"}), 500

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
