from flask import Flask, jsonify, request
from researcher import ResearcherAgent

app = Flask(__name__)
researcher_agent = ResearcherAgent()

@app.route('/research', methods=['POST'])
def research():
    try:
        data = request.json  # Extract JSON payload
        print(f"Received payload: {data}")  # Log the incoming data for debugging

        if not data:
            return jsonify({"error": "Invalid or missing JSON payload."}), 400

        stock_ticker = data.get("stock_ticker", "")
        print(f"Extracted stock ticker: {stock_ticker}")  # Log the extracted stock ticker

        if not stock_ticker:
            return jsonify({"error": "Stock ticker is required."}), 400

        # Call the ResearcherAgent
        result = researcher_agent.handle_task({"stock_ticker": stock_ticker})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
