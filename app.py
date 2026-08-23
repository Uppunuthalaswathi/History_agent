"""
Flask Application for Computing History Agent Client.

This application provides:
1. Web interface for the Computing History AI Agent
2. Chat functionality
3. Conversation reset
4. Upload PDF/TXT/DOCX documents
5. Automatic document chunking, embedding, and Azure AI Search indexing
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import markdown
import bleach

from agent_client import AgentClient
from rag.upload_ingestion import ingest_uploaded_file


# ---------------------------------------------------------
# Flask Application
# ---------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------
# Upload Configuration
# ---------------------------------------------------------

UPLOAD_FOLDER = Path("rag/documents")

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
}

# Maximum upload size: 20 MB
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# ---------------------------------------------------------
# Markdown / HTML Security
# ---------------------------------------------------------

def _set_external_link_attributes(attrs, new=False):
    """Force safe external link attributes for rendered markdown links."""

    href_key = (None, "href")
    href_value = attrs.get(href_key, "")

    if isinstance(href_value, str) and href_value.startswith(
        ("http://", "https://")
    ):
        attrs[(None, "target")] = "_blank"
        attrs[(None, "rel")] = "noopener noreferrer nofollow"

    return attrs


def render_markdown_to_safe_html(text: str) -> str:
    """Convert markdown to safe HTML for display in chat bubbles."""

    raw_html = markdown.markdown(
        text,
        extensions=[
            "extra",
            "sane_lists",
            "nl2br",
        ],
    )

    allowed_tags = [
        "p",
        "br",
        "hr",
        "blockquote",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "strong",
        "em",
        "code",
        "pre",
        "a",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
    ]

    allowed_attrs = {
        "a": [
            "href",
            "title",
            "target",
            "rel",
        ],
        "code": [
            "class",
        ],
    }

    safe_html = bleach.clean(
        raw_html,
        tags=allowed_tags,
        attributes=allowed_attrs,
        protocols=[
            "http",
            "https",
            "mailto",
        ],
        strip=True,
    )

    # Linkify plain URLs while leaving code blocks untouched.
    safe_html = bleach.linkify(
        safe_html,
        skip_tags=[
            "pre",
            "code",
        ],
        callbacks=[
            _set_external_link_attributes
        ],
    )

    return safe_html


# ---------------------------------------------------------
# Initialize Agent Client
# ---------------------------------------------------------

try:
    agent = AgentClient()

    print("✓ Agent client initialized successfully.")

except Exception as e:

    print(
        f"Warning: Failed to initialize agent client: {e}"
    )

    agent = None


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------


@app.route("/")
def index():
    """Render the main chat interface."""

    return render_template(
        "index.html"
    )


# ---------------------------------------------------------
# Chat Route
# ---------------------------------------------------------


@app.route(
    "/chat",
    methods=["POST"]
)
def chat():
    """Handle chat messages from the user."""

    if not agent:

        return jsonify(
            {
                "error": (
                    "Agent client not initialized. "
                    "Check your environment variables."
                )
            }
        ), 500

    data = request.get_json(
        silent=True
    ) or {}

    user_message = data.get(
        "message",
        ""
    ).strip()

    if not user_message:

        return jsonify(
            {
                "error": "Message is required"
            }
        ), 400

    # Validate message length.
    if len(user_message) > 10000:

        return jsonify(
            {
                "error": "Message too long"
            }
        ), 400

    try:

        print(
            f"User question: {user_message}"
        )

        response = agent.send_message(
            user_message
        )

        response_html = (
            render_markdown_to_safe_html(
                response
            )
        )

        return jsonify(
            {
                "response": response,
                "response_html": response_html,
            }
        )

    except Exception as e:

        print(
            "Error while processing "
            f"chat request: {e}"
        )

        return jsonify(
            {
                "error": (
                    "Failed to process your message."
                )
            }
        ), 500


# ---------------------------------------------------------
# Upload Route
# ---------------------------------------------------------


@app.route(
    "/upload",
    methods=["POST"]
)
def upload_file():
    """
    Upload a PDF/TXT/DOCX document.

    Workflow:

    Browser
        ↓
    Flask
        ↓
    Save document
        ↓
    Extract text
        ↓
    Chunk document
        ↓
    Generate embeddings
        ↓
    Azure AI Search
    """

    try:

        # ---------------------------------------------
        # Check whether a file was sent
        # ---------------------------------------------

        if "file" not in request.files:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "No file was uploaded."
                    ),
                }
            ), 400

        file = request.files["file"]


        # ---------------------------------------------
        # Check filename
        # ---------------------------------------------

        if not file.filename:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "No file was selected."
                    ),
                }
            ), 400


        # ---------------------------------------------
        # Secure filename
        # ---------------------------------------------

        filename = secure_filename(
            file.filename
        )


        if not filename:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Invalid filename."
                    ),
                }
            ), 400


        # ---------------------------------------------
        # Check extension
        # ---------------------------------------------

        extension = Path(
            filename
        ).suffix.lower()


        if extension not in ALLOWED_EXTENSIONS:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Unsupported file type. "
                        "Please upload PDF, TXT, "
                        "or DOCX."
                    ),
                }
            ), 400


        # ---------------------------------------------
        # Save the upload only for the duration of ingestion.  The
        # searchable copy is stored in the existing Azure AI Search index.
        # ---------------------------------------------

        temporary_file = tempfile.NamedTemporaryFile(
            suffix=extension,
            prefix="computing-historian-",
            dir=UPLOAD_FOLDER,
            delete=False,
        )
        save_path = Path(temporary_file.name)
        temporary_file.close()
        try:
            file.save(save_path)
            print(f"Uploaded file saved: {save_path}")

            # ---------------------------------------------
            # Ingest into Azure AI Search
            # ---------------------------------------------

            print(
                f"Starting RAG ingestion for: "
                f"{filename}"
            )

            result = ingest_uploaded_file(
                str(save_path),
                original_filename=filename,
            )
        finally:
            # Do not retain a second local knowledge base or users' uploads.
            save_path.unlink(missing_ok=True)


        # ---------------------------------------------
        # Return success
        # ---------------------------------------------

        return jsonify(
            {
                "success": True,
                "filename": result["filename"],
                "chunks": result["chunks"],
                "message": (
                    "Document uploaded and "
                    "indexed successfully."
                ),
            }
        )


    except Exception as e:

        print(
            "Upload error:",
            repr(e)
        )

        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        ), 500


# ---------------------------------------------------------
# Reset Conversation
# ---------------------------------------------------------


@app.route(
    "/reset",
    methods=["POST"]
)
def reset():
    """Reset the conversation history."""

    if agent:

        try:

            agent.reset_conversation()

        except Exception as e:

            print(
                "Error while resetting "
                f"conversation: {e}"
            )

            return jsonify(
                {
                    "error": (
                        "Failed to reset conversation."
                    )
                }
            ), 500

    return jsonify(
        {
            "status": "success"
        }
    )


# ---------------------------------------------------------
# File Upload Size Error
# ---------------------------------------------------------


@app.errorhandler(413)
def file_too_large(error):
    """Handle files larger than MAX_CONTENT_LENGTH."""

    return jsonify(
        {
            "success": False,
            "error": (
                "File is too large. "
                "Maximum allowed size is 20 MB."
            ),
        }
    ), 413


# ---------------------------------------------------------
# Run Flask Application
# ---------------------------------------------------------


if __name__ == "__main__":

    # Render provides the PORT environment variable.
    # Locally, it uses port 5000 if PORT is not set.

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        f"Starting Flask server "
        f"on 0.0.0.0:{port}"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
