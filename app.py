from flask import Flask, jsonify, request
from researcher import ResearcherAgent

app = Flask(__name__)
researcher_agent = ResearcherAgent()

@app.route('/research', methods=['POST'])
def research():
    try:
        data = request.json  # Make sure this is not `None`
        stock_ticker = data.get("stock_ticker", "")  # Ensure this matches your payload key

        if not stock_ticker:
            return jsonify({"error": "Stock ticker is required."}), 400

        # Process the request
        result = researcher_agent.handle_task({"stock_ticker": stock_ticker})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Note: Do NOT include app.run() for production.
