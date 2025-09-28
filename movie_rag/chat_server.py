"""chat_server

A Flask-based server which provides a simple chat interface for querying movies.
It uses an agent with three tools to access the data:
  - SQL: for structured queries
  - Vector Store: for similarity search
  - Graph: to traverse a knowledge graph (optional)

The chat history is optionally kept and used as part of the prompt.
"""

import sys, os
from flask import Flask, render_template, request, jsonify

# Allow running as a module (python -m movie_rag.chat_server) and as a script
try:
    from .chatter import MovieChatter
    from . import utils
except ImportError:  # direct script run from package directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from movie_rag.chatter import MovieChatter
    from movie_rag import utils


def define_routes(server):
    @server.app.route('/movie-rag')
    def index():
        return render_template('index.html')

    @server.app.route('/chat', methods=['POST'])
    def chat():
        user_input = request.form['user_input']
        response = server.movie_chatter.chat(user_input)

        return jsonify({'response': response})

    @server.app.route('/forget', methods=['POST'])
    def forget():
        response = server.movie_chatter.forget()
        return jsonify({'response': response})

    return


class ChatServer:
    def __init__(self, argv):
        self.config = utils.get_config(argv)
        # Ensure Flask can locate templates relative to this file
        self.app = Flask(
            __name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates')
        )
        self.movie_chatter = MovieChatter(**self.config)

        return

    def run(self):
        host = self.config.get('host', '127.0.0.1')
        port = self.config.get('port', 5000)
        debug = self.config.get('debug', True)
        self.app.run(host=host, port=port, debug=debug)

        return


if __name__ == '__main__':
    chat_server = ChatServer(sys.argv[1:])
    define_routes(chat_server)
    chat_server.run()
