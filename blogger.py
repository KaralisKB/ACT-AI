from crewai.agent import Agent
import requests

class CrewAIBloggerAgent(Agent):
    def __init__(self, ngrok_url):
        super().__init__(
            role="Blogger",
            goal="Generate concise and engaging summaries of recommendations.",
            backstory="A content creation AI agent designed to summarize recommendations in a reader-friendly format."
        )
        self.ngrok_url = ngrok_url

    def handle_task(self, task_input):
        recommendation = task_input.get("recommendation", "")
        rationale = task_input.get("rationale", "")
        if not recommendation or not rationale:
            return {"error": "Both recommendation and rationale are required."}

        try:
            # Send data to the locally hosted Blogger API
            response = requests.post(
                f"{self.ngrok_url}/blogger",
                json={"recommendation": recommendation, "rationale": rationale},
                timeout=30
            )
            if response.status_code != 200:
                return {"error": f"Blogger API Error: {response.text}"}
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to connect to Blogger API: {str(e)}"}