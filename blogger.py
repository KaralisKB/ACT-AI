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

    def handle_task(self, recommender_result):
        try:
            recommendation = recommender_result.get("recommendation", None)
            rationale = recommender_result.get("rationale", None)

            # Validate inputs
            if not recommendation or not rationale:
                raise ValueError("Both recommendation and rationale are required.")

            # Build a concise summary
            summary = f"The recommendation is to **{recommendation}**. This is because: {rationale}"

            # Return the summarized blog post
            return {"blog_post": summary}
        except Exception as e:
            return {"error": f"Blogger Agent Error: {str(e)}"}
