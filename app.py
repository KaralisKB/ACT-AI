from flask import Flask, request, jsonify
from agents import Researcher, Accountant, Recommender, Blogger

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json  # Expect payload with stock symbol, name, etc.

    # Extract inputs
    stock_symbol = data.get('symbol')
    stock_name = data.get('name')

    # Run agents
    researcher = Researcher(stock_symbol, stock_name)
    research_data = researcher.get_data()

    accountant = Accountant(research_data)
    ratios = accountant.calculate_ratios()

    recommender = Recommender(research_data, ratios)
    decision = recommender.make_decision()

    blogger = Blogger(research_data, ratios, decision)
    report = blogger.format_report()

    # Return recommendation and report
    return jsonify({
        "recommendation": decision,
        "report": report
    })
