from flask import Flask, request, jsonify
from recommender import RecommenderAgent
from accountant import CrewAIAccountantAgent
from blogger import CrewAIBloggerAgent
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize agents
recommender_agent = RecommenderAgent()
accountant_agent = CrewAIAccountantAgent()
blogger_agent = CrewAIBloggerAgent()

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Parse input data
        input_data = request.json
        if not input_data:
            return jsonify({"error": "No input data provided."}), 400

        # Step 1: Call the Accountant Agent
        accountant_response = accountant_agent.handle_task({"financial_data": input_data.get("financial_data", {})})
        if "error" in accountant_response:
            return jsonify({"error": f"Accountant Agent Error: {accountant_response['error']}"}), 500

        # Step 2: Call the Recommender Agent
        recommender_response = recommender_agent.handle_task(
            researcher_data=input_data.get("researcher_data", {}),
            accountant_data=accountant_response
        )
        if "error" in recommender_response:
            return jsonify({"error": f"Recommender Agent Error: {recommender_response['error']}"}), 500

        # Step 3: Call the Blogger Agent
        blogger_response = blogger_agent.handle_task(recommender_response)
        if "error" in blogger_response:
            return jsonify({"error": f"Blogger Agent Error: {blogger_response['error']}"}), 500

        # Return the final blog post
        return jsonify(blogger_response), 200

    except Exception as e:
        return jsonify({"error": f"Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    # Run the app (development server for testing purposes)
    app.run(host="0.0.0.0", port=5000, debug=True)
