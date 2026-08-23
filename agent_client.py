import os
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

from rag.search import ComputingHistorySearch


load_dotenv()

logger = logging.getLogger(__name__)


class AgentClient:

    def __init__(self):

        # =========================================================
        # Foundry Agent configuration
        # =========================================================

        self.agent_endpoint = os.getenv("AGENT_ENDPOINT")

        # Fallback to the project endpoint if AGENT_ENDPOINT
        # is not present.
        if not self.agent_endpoint:
            self.agent_endpoint = os.getenv(
                "AZURE_AI_PROJECT_ENDPOINT"
            )

        if not self.agent_endpoint:
            raise ValueError(
                "AGENT_ENDPOINT or AZURE_AI_PROJECT_ENDPOINT "
                "not found"
            )

        self.agent_endpoint = self.agent_endpoint.rstrip("/")

        print(
            "Using Foundry endpoint:",
            self.agent_endpoint
        )

        # =========================================================
        # Foundry Project Client
        # =========================================================

        credential = DefaultAzureCredential()

        project_client = AIProjectClient(
            endpoint=self.agent_endpoint,
            credential=credential,
        )

        self.client = project_client.get_openai_client()

        # =========================================================
        # RAG Search
        # =========================================================

        print("Initializing Computing History RAG...")

        try:
            self.rag = ComputingHistorySearch()

            print(
                "✓ Computing History RAG initialized."
            )

        except Exception as exc:

            logger.exception(
                "Failed to initialize RAG"
            )

            raise RuntimeError(
                f"RAG initialization failed: {exc}"
            ) from exc

        # =========================================================
        # Conversation memory
        # =========================================================

        self.conversation_history: List[
            Dict[str, Any]
        ] = []

        self.max_history = 3

    # =============================================================
    # RAG CONTEXT
    # =============================================================

    def _retrieve_context(
        self,
        user_message: str,
        top_k: int = 5,
    ) -> str:
        """
        Retrieve relevant Computing History information
        from Azure AI Search.
        """

        try:

            results = self.rag.search(
                user_message,
                top_k=top_k,
                mode="hybrid",
            )

        except Exception as exc:

            logger.exception(
                "RAG search failed"
            )

            print(
                f"⚠ RAG search failed: {exc}"
            )

            return ""

        if not results:
            print(
                "⚠ No relevant RAG documents found."
            )

            return ""

        context_parts = []

        for position, result in enumerate(
            results,
            start=1,
        ):

            context_parts.append(
                f"""
SOURCE {position}
Title: {result.title}
Source: {result.source}
Chunk ID: {result.chunk_id}

Content:
{result.content}
""".strip()
            )

        context = "\n\n".join(
            context_parts
        )

        print(
            f"✓ Retrieved {len(results)} "
            f"relevant RAG chunks."
        )

        return context

    # =============================================================
    # SEND MESSAGE
    # =============================================================

    def send_message(
        self,
        user_message: str,
    ) -> str:

        if not user_message or not user_message.strip():
            return "Please enter a question."

        # =========================================================
        # Retrieve relevant information from Azure AI Search
        # =========================================================

        print(
            "\nSearching Computing History knowledge base..."
        )

        context = self._retrieve_context(
            user_message,
            top_k=5,
        )

        # =========================================================
        # Build grounded prompt
        # =========================================================

        if context:

            grounded_message = f"""
You are Computing Historian, an AI assistant specialized
in the history of computing and artificial intelligence.

Use the following retrieved information from the knowledge base
(which can include the user's uploaded document) to answer the question.

IMPORTANT RULES:

1. Prefer the retrieved knowledge-base information. If it is from an
   uploaded document, answer from that document rather than substituting
   a generic computing-history answer.
2. Do not invent historical facts.
3. If the retrieved context does not contain enough
   information to answer the question, clearly say that
   the knowledge base does not contain enough information.
4. You may use your general knowledge to provide helpful
   context, but do not contradict the retrieved sources.
5. Give a clear and concise answer.
6. When appropriate, mention the source or historical
   person/event from the retrieved context.

============================================================
RETRIEVED COMPUTING HISTORY CONTEXT
============================================================

{context}

============================================================
END RETRIEVED CONTEXT
============================================================

USER QUESTION:
{user_message}
""".strip()

        else:

            grounded_message = f"""
You are Computing Historian, an AI assistant specialized
in the history of computing and artificial intelligence.

The knowledge base did not return relevant information
for this question.

Answer carefully using your general knowledge, and do
not invent facts.

USER QUESTION:
{user_message}
""".strip()

        # =========================================================
        # Conversation history
        # =========================================================

        self.conversation_history.append({
            "role": "user",
            "content": grounded_message,
        })

        try:

            # =====================================================
            # Foundry Agent
            # =====================================================

            response = self.client.responses.create(
                input=self.conversation_history,
                extra_body={
                    "agent_reference": {
                        "name": "computing-historian",
                        "version": "1",
                        "type": "agent_reference",
                    }
                },
            )

            assistant_message = response.output_text

            # =====================================================
            # Store assistant response
            # =====================================================

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message,
            })

            # =====================================================
            # Limit conversation memory
            # =====================================================

            if (
                len(self.conversation_history)
                > self.max_history * 2
            ):

                self.conversation_history = (
                    self.conversation_history[
                        -self.max_history * 2:
                    ]
                )

            return assistant_message

        except Exception as exc:

            logger.exception(
                "Agent error"
            )

            return f"Error: {str(exc)}"

    # =============================================================
    # RESET CONVERSATION
    # =============================================================

    def reset_conversation(self):

        self.conversation_history = []

        print(
            "✓ Conversation history reset."
        )
