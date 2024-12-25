from flask import Flask, jsonify, request
from researcher import ResearcherAgent
from accountant import AccountantAgent

app = Flask(__name__)

# Initialize the agents
researcher_agent = ResearcherAgent()
accountant_agent = AccountantAgent()

@app.route('/analyze', methods=['POST'])
def analyze_stock():
    try:
        # Parse the incoming JSON payload
        data = request.json
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload."}), 400

        stock_ticker = data.get("stock_ticker", "")
        if not stock_ticker:
            return jsonify({"error": "Stock ticker is required."}), 400

        # Step 1: Call the ResearcherAgent
        print(f"Running ResearcherAgent for stock: {stock_ticker}")
        researcher_result = researcher_agent.handle_task({"stock_ticker": stock_ticker})
        if isinstance(researcher_result, str) and "Error" in researcher_result:
            return jsonify({"error": f"ResearcherAgent error: {researcher_result}"}), 500

        # Step 2: Call the AccountantAgent
        print(f"Running AccountantAgent for stock: {stock_ticker}")
        accountant_result = accountant_agent.handle_task(researcher_result)
        if isinstance(accountant_result, str) and "Error" in accountant_result:
            return jsonify({"error": f"AccountantAgent error: {accountant_result}"}), 500

        # Step 3: Combine the results and return
        response = {
            "researcher_data": researcher_result,
            "accountant_data": accountant_result
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
