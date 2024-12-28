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
        start_time = time.time()
        logger.debug("Received analysis request")

        data = request.json

        if not data:
            return jsonify({"error": "Invalid or missing JSON payload."}), 400

        stock_ticker = data.get("stock_ticker", "")
        if not stock_ticker:
            return jsonify({"error": "Stock ticker is required."}), 400

        # Step 1: Use the Researcher Agent
        researcher_start = time.time()
        researcher_result = researcher_agent.handle_task({"stock_ticker": stock_ticker})
        researcher_time = time.time() - researcher_start
        logger.debug(f"Researcher Agent completed in {researcher_time:.2f} seconds")

        if "error" in researcher_result:
            return jsonify({"error": f"Researcher Agent Error: {researcher_result['error']}"}), 500

        # Step 2: Use the Accountant Agent
        accountant_start = time.time()
        accountant_result = accountant_agent.handle_task({"financial_data": researcher_result["financial_data"]})
        accountant_time = time.time() - accountant_start
        logger.debug(f"Accountant Agent completed in {accountant_time:.2f} seconds")

        if "error" in accountant_result:
            return jsonify({"error": f"Accountant Agent Error: {accountant_result['error']}"}), 500

        # Step 3: Use the Recommender Agent
        recommender_start = time.time()
        recommender_result = recommender_agent.handle_task(researcher_result, accountant_result)
        recommender_time = time.time() - recommender_start
        logger.debug(f"Recommender Agent completed in {recommender_time:.2f} seconds")

        if "error" in recommender_result:
            return jsonify({"error": f"Recommender Agent Error: {recommender_result['error']}"}), 500

        # Step 4: Use the Blogger Agent
        blogger_start = time.time()
        blogger_result = blogger_agent.handle_task({
            "recommendation": recommender_result.get("recommendation"),
            "rationale": recommender_result.get("rationale")
        })
        blogger_time = time.time() - blogger_start
        logger.debug(f"Blogger Agent completed in {blogger_time:.2f} seconds")

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

        total_time = time.time() - start_time
        logger.debug(f"Total analysis process completed in {total_time:.2f} seconds")
        return jsonify(combined_result)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
