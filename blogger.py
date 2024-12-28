from crewai.agent import Agent
import requests
import os

NGROK_URL = os.getenv("LOCAL_NGROK_URL")


class CrewAIBloggerAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Blogger",
            goal="Generate concise and consistent summaries of recommendations.",
            backstory="An AI designed to produce succinct summaries of stock recommendations for clarity."
        )

    def handle_task(self, recommender_result):
        try:
            # Extract recommendation and rationale
            recommendation = recommender_result.get("recommendation", None)
            rationale = recommender_result.get("rationale", None)

            # Validate inputs
            if not recommendation or not rationale:
                raise ValueError("Both recommendation and rationale are required.")

            # Build a concise summary
            summary = f"The recommendation is to **{recommendation}**. Reason: {rationale}"

            # Return the summarized blog post
            return {"recommendation": recommendation, "reasoning": rationale}
        except Exception as e:
            return {"error": f"Blogger Agent Error: {str(e)}"}