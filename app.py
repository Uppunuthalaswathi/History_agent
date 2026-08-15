"""
Flask Application for Computing History Agent Client.

This is the main Flask application that provides a web interface
for interacting with the Computing History agent.
"""

import os

from flask import Flask, render_template, request, jsonify
import markdown
import bleach

from agent_client import AgentClient


app = Flask(__name__)


def _set_external_link_attributes(attrs, new=False):
    """Force safe external link attributes for rendered markdown links."""
    href_key = (None, 'href')
    href_value = attrs.get(href_key, '')

    if isinstance(href_value, str) and href_value.startswith(
        ('http://', 'https://')
    ):
        attrs[(None, 'target')] = '_blank'
        attrs[(None, 'rel')] = 'noopener noreferrer nofollow'

    return attrs


def render_markdown_to_safe_html(text: str) -> str:
    """Convert markdown to safe HTML for display in chat bubbles."""

    raw_html = markdown.markdown(
        text,
        extensions=['extra', 'sane_lists', 'nl2br']
    )

    allowed_tags = [
        'p', 'br', 'hr', 'blockquote',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li',
        'strong', 'em', 'code', 'pre',
        'a',
        'table', 'thead', 'tbody', 'tr', 'th', 'td'
    ]

    allowed_attrs = {
        'a': ['href', 'title', 'target', 'rel'],
        'code': ['class']
    }

    safe_html = bleach.clean(
        raw_html,
        tags=allowed_tags,
        attributes=allowed_attrs,
        protocols=['http', 'https', 'mailto'],
        strip=True
    )

    # Linkify plain URLs while leaving code blocks untouched.
    safe_html = bleach.linkify(
        safe_html,
        skip_tags=['pre', 'code'],
        callbacks=[_set_external_link_attributes]
    )

    return safe_html


# ---------------------------------------------------------
# Initialize the Agent Client
# ---------------------------------------------------------

try:
    agent = AgentClient()
except Exception as e:
    print(f"Warning: Failed to initialize agent client: {e}")
    agent = None


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from the user."""

    if not agent:
        return jsonify({
            'error': 'Agent client not initialized. Check your environment variables.'
        }), 500

    data = request.get_json(silent=True) or {}

    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({
            'error': 'Message is required'
        }), 400

    # Validate message length to prevent abuse
    if len(user_message) > 10000:
        return jsonify({
            'error': 'Message too long'
        }), 400

    try:
        response = agent.send_message(user_message)

        response_html = render_markdown_to_safe_html(response)

        return jsonify({
            'response': response,
            'response_html': response_html
        })

    except Exception as e:
        print(f"Error while processing chat request: {e}")

        return jsonify({
            'error': 'Failed to process your message.'
        }), 500


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the conversation history."""

    if agent:
        try:
            agent.reset_conversation()
        except Exception as e:
            print(f"Error while resetting conversation: {e}")

            return jsonify({
                'error': 'Failed to reset conversation.'
            }), 500

    return jsonify({
        'status': 'success'
    })


# ---------------------------------------------------------
# Run Flask Application
# ---------------------------------------------------------

if __name__ == '__main__':

    # Render provides the PORT environment variable.
    # Locally, it will use port 5000 if PORT is not set.
    port = int(os.environ.get('PORT', 5000))

    print(f"Starting Flask server on 0.0.0.0:{port}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
