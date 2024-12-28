from crewai.agent import Agent
import requests

class CrewAIAccountantAgent(Agent):
    def __init__(self, ngrok_url):
        super().__init__(
            role="Accountant",
            goal="Perform financial calculations and provide insights.",
            backstory="A financial expert AI agent capable of analyzing financial data and producing key metrics."
        )
        self.ngrok_url = ngrok_url

    def handle_task(self, task_input):
        financial_data = task_input.get("financial_data", {})
        if not financial_data:
            return {"error": "No financial data provided."}

        try:
            # Send data to the locally hosted Accountant API
            response = requests.post(
                f"{self.ngrok_url}/accountant",
                json={"financial_data": financial_data},
                timeout=30
            )
            if response.status_code != 200:
                return {"error": f"Accountant API Error: {response.text}"}
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to connect to Accountant API: {str(e)}"}
