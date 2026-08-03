import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

logger = logging.getLogger(__name__)


class AgentClient:

    def __init__(self):

        self.agent_endpoint = os.getenv("AGENT_ENDPOINT")

        if not self.agent_endpoint:
            raise ValueError("AGENT_ENDPOINT not found")

        self.agent_endpoint = self.agent_endpoint.rstrip("/")

        print("Using endpoint:", self.agent_endpoint)

        # ✅ CORRECT FOUNDARY CLIENT
        project_client = AIProjectClient(
            endpoint=self.agent_endpoint,
            credential=DefaultAzureCredential(),
        )

        self.client = project_client.get_openai_client()

        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = 3

    def send_message(self, user_message: str) -> str:

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:

            response = self.client.responses.create(
                input=self.conversation_history,
                extra_body={
                    "agent_reference": {
                        "name": "computing-historian",
                        "version": "1",
                        "type": "agent_reference"
                    }
                }
            )

            assistant_message = response.output_text

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]

            return assistant_message

        except Exception as e:
            logger.exception("Agent error")
            return f"Error: {str(e)}"

    def reset_conversation(self):
        self.conversation_history = []