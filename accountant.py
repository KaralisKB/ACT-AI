from crewai.agent import Agent
import requests
import os

NGROK_URL = os.getenv("LOCAL_NGROK_URL")

class CrewAIAccountantAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Accountant",
            goal="Perform financial calculations and provide insights.",
            backstory="A financial expert AI agent capable of analyzing financial data and producing key metrics."
        )
        

    def handle_task(self, task_input):
        financial_data = task_input.get("financial_data", {})
        if not financial_data:
            return {"error": "No financial data provided."}

        try:
            # Send data to the locally hosted Accountant API
            response = requests.post(
                f"{NGROK_URL}/accountant",
                json={"financial_data": financial_data},
                timeout=60
            )
            if response.status_code != 200:
                return {"error": f"Accountant API Error: {response.text}"}
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to connect to Accountant API: {str(e)}"}
