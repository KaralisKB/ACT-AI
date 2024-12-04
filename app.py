from flask import Flask, request, jsonify
from agents import Researcher, Accountant, Recommender, Blogger

app = Flask(__name__)

@app.route('/research', methods=['POST'])
def research():
    try:
        print("Request received")  # Add this log
        data = request.json
        print("Request data:", data)  # Log request data
        stock_ticker = data.get("stock_ticker", "")

        if not stock_ticker:
            return jsonify({"error": "Stock ticker is required."}), 400

        # Call the ResearcherAgent
        result = researcher_agent.handle_task({"stock_ticker": stock_ticker})
        print("Research result:", result)  # Log result
        return jsonify(result)
    except Exception as e:
        print("Error occurred:", str(e))  # Log error
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500