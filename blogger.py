from crewai.agent import Agent
import requests
import os

NGROK_URL = os.getenv("LOCAL_NGROK_URL")


class CrewAIBloggerAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Blogger",
            goal="Generate concise and engaging summaries of recommendations.",
            backstory="A content creation AI agent designed to summarize recommendations in a reader-friendly format."
        )

    def handle_task(self, task_input):
        recommendation = task_input.get("recommendation", "")
        rationale = task_input.get("rationale", "")
        if not recommendation or not rationale:
            return {"error": "Both recommendation and rationale are required."}

        try:
            # Send data to the locally hosted Blogger API
            response = requests.post(
                f"{NGROK_URL}/blogger",
                json={"recommendation": recommendation, "rationale": rationale},
                timeout=60
            )
            if response.status_code != 200:
                return {"error": f"Blogger API Error: {response.text}"}
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to connect to Blogger API: {str(e)}"}
